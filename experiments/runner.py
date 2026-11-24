"""Shared, paired LOL training and evaluation for the independent CIDNet experiments."""
import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from loss.losses import EdgeLoss, PerceptualLoss, SSIM
from measure import calculate_psnr, calculate_ssim
from net.CIDNet import CIDNet


def image_index(folder):
    images = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            key = re.sub(r'^(low|normal|high)[_-]?(?=\d)', '', path.stem.lower())
            if key in images:
                raise ValueError(f'Duplicate pair identifier {key} in {folder}')
            images[key] = path
    return images


class Pairs(Dataset):
    def __init__(self, root, crop=0):
        low_dir, gt_dir = ('Low', 'Normal') if (root / 'Low').is_dir() else ('low', 'high')
        low, gt = image_index(root / low_dir), image_index(root / gt_dir)
        if not low or low.keys() != gt.keys():
            raise ValueError(f'Empty or unmatched pairs in {root}, mismatches: {sorted(low.keys() ^ gt.keys())[:5]}')
        self.pairs = [(low[k], gt[k]) for k in sorted(low)]
        self.crop = crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        paths = self.pairs[index]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        w, h = images[0].size
        if images[1].size != (w, h):
            raise ValueError(f'Size mismatch for {paths}')
        if self.crop:
            if min(h, w) < self.crop:
                raise ValueError(f'{paths[0]} smaller than --crop {self.crop}')
            x = random.randint(0, w - self.crop)
            y = random.randint(0, h - self.crop)
            images = [im.crop((x, y, x + self.crop, y + self.crop)) for im in images]
            for flip in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < 0.5:
                    images = [im.transpose(flip) for im in images]
        return to_tensor(images[0]), to_tensor(images[1])


def noisy_views(low, std):
    return [(low + std * torch.randn_like(low)).clamp(0, 1) for _ in range(2)]


class Objective(nn.Module):
    """RGB and HVI reconstruction, with the original VGG feature objective."""
    def __init__(self, perceptual_weight):
        super().__init__()
        self.edge = EdgeLoss(loss_weight=50)
        self.ssim = SSIM(weight=0.5)
        self.weight = perceptual_weight
        self.vgg = None
        if self.weight:
            layers = dict.fromkeys(('conv1_2', 'conv2_2', 'conv3_4', 'conv4_4'), 1)
            self.vgg = PerceptualLoss(layers, criterion='mse')

    def forward(self, model, output, target):
        rec = output.new_zeros(())
        perceptual = output.new_zeros(())
        for pred, gt in ((output, target), (model.HVIT(output), model.HVIT(target))):
            rec = rec + F.l1_loss(pred, gt) + self.edge(pred, gt) + self.ssim(pred, gt)
            if self.vgg is not None:
                perceptual = perceptual + self.weight * self.vgg(pred, gt)[0]
        return rec, perceptual


def standard_backward(model, low, target, objective, args):
    rec, perceptual = objective(model, model(low), target)
    loss = rec + perceptual
    loss.backward()
    return loss.detach()


@torch.inference_mode()
def evaluate(model, pairs, device, lpips_net, alpha):
    model.eval()
    totals = {'psnr': 0., 'ssim': 0.}
    if lpips_net is not None:
        totals['lpips'] = 0.
    for low, target in DataLoader(pairs, batch_size=1):
        h, w = low.shape[-2:]
        if min(h, w) < 11:
            raise ValueError('SSIM needs images of at least 11 x 11 pixels')
        low = F.pad(low.to(device), (0, -w % 8, 0, -h % 8), mode='reflect')
        pred = (alpha * model(low)[..., :h, :w]).clamp(0, 1)
        # Match torchvision ToPILImage in the original 8-bit evaluation.
        pred = pred.mul(255).byte()
        gt = target.mul(255).round().byte()
        pred_rgb = pred[0].permute(1, 2, 0).cpu().numpy()
        gt_rgb = gt[0].permute(1, 2, 0).numpy()
        totals['psnr'] += float(calculate_psnr(pred_rgb, gt_rgb))
        totals['ssim'] += float(calculate_ssim(pred_rgb, gt_rgb))
        if lpips_net is not None:
            totals['lpips'] += lpips_net(pred.float() / 127.5 - 1, gt.to(device).float() / 127.5 - 1).item()
    return {key: value / len(pairs) for key, value in totals.items()}


def load_model(path, model, variant):
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint['variant'] != variant:
        raise ValueError(f"Expected {variant}, found {checkpoint['variant']} in {path}")
    model.load_state_dict(checkpoint['model'])
    return checkpoint


def save_checkpoint(state, path):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def main(variant, model_class, backward, add_options=None):
    parser = argparse.ArgumentParser(description=f'HVI experiment {variant}')
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('--train-dir', type=Path)
    parser.add_argument('--val-dir', type=Path, help='Paired validation or evaluation directory')
    parser.add_argument('--out', type=Path, help='Training run directory')
    parser.add_argument('--checkpoint', type=Path, help='Experiment checkpoint for evaluation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--init', type=Path, help='Original CIDNet state_dict for warm start')
    group.add_argument('--resume', action='store_true', help='Resume latest.pt from --out')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--perceptual-weight', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1., help='Fixed output gain for evaluation')
    parser.add_argument('--lpips', action='store_true', help='Include pretrained AlexNet LPIPS in evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    if add_options:
        add_options(parser)
    args = parser.parse_args()
    if args.command == 'eval' and (not args.val_dir or not args.checkpoint):
        parser.error('eval needs --val-dir and --checkpoint')
    if args.command == 'train' and not args.train_dir:
        parser.error('train needs --train-dir')
    if min(args.epochs, args.batch_size, args.eval_every, args.crop) < 1 or args.crop % 8:
        parser.error('positive epochs, batch size and interval required; crop must be a positive multiple of eight')
    if args.perceptual_weight < 0 or args.lr <= 0 or args.alpha <= 0 or args.workers < 0:
        parser.error('invalid loss, learning rate, evaluation gain or worker count')
    if variant == 'd' and args.command == 'train' and args.perceptual_weight == 0:
        parser.error('D requires a nonzero perceptual loss')
    if variant == 'i' and not 0 < args.hard_fraction <= 1:
        parser.error('--hard-fraction must be in (0, 1]')
    weights = ('consistency_weight', 'noise_std', 'band_weight', 'chroma_weight', 'hard_weight')
    if any(not math.isfinite(getattr(args, key, 0)) or getattr(args, key, 0) < 0 for key in weights):
        parser.error('noise and auxiliary loss weights must be nonnegative')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = model_class().to(device)
    print(f'{variant}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    validation = Pairs(args.val_dir) if args.val_dir else None
    lpips_net = None
    if args.lpips:
        import lpips
        lpips_net = lpips.LPIPS(net='alex').to(device).eval()
    if args.command == 'eval':
        load_model(args.checkpoint, model, variant)
        print(json.dumps(evaluate(model, validation, device, lpips_net, args.alpha)))
        return

    train = Pairs(args.train_dir, args.crop)
    out = args.out or Path('results/experiments') / args.train_dir.parent.name / variant
    out.mkdir(parents=True, exist_ok=True)
    latest = out / 'latest.pt'
    if not args.resume and any(out.iterdir()):
        parser.error(f'{out} already contains files; use --resume or another --out')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.init:
        # Validate all baseline keys/shapes before accepting missing variant-specific tensors.
        state = torch.load(args.init, map_location='cpu')
        CIDNet().load_state_dict(state, strict=True)
        model.load_state_dict(state, strict=False)
    objective = Objective(args.perceptual_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    history = []
    best, start = -math.inf, 0
    if args.resume:
        state = load_model(latest, model, variant)
        for key in ('epochs', 'train_dir', 'crop', 'batch_size', 'perceptual_weight', 'alpha', 'val_dir',
                    'seed', 'lr', 'noise_std', 'bands', 'band_mode', 'band_weight',
                    'consistency_weight', 'disable_projection', 'chroma_weight', 'hard_weight', 'hard_fraction'):
            if config.get(key) != state['config'].get(key):
                parser.error(f'--resume requires the same {key}')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best = state['epoch'], state['best_psnr']
        history = state['history']
        # The checkpoint is authoritative if interrupted between save and log append.
        log_tmp = out / 'metrics.tmp'
        log_tmp.write_text(''.join(json.dumps(row) + '\n' for row in history))
        os.replace(log_tmp, out / 'metrics.jsonl')
    else:
        with (out / 'config.json').open('x') as stream:
            json.dump({'variant': variant, **config}, stream, indent=2)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        sampler_rng = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            generator=sampler_rng, pin_memory=device.type == 'cuda')
        total = 0.
        for low, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = backward(model, low.to(device), target.to(device), objective, args)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * low.shape[0]
        record = {'epoch': epoch, 'loss': total / len(train), 'lr': optimizer.param_groups[0]['lr']}
        scheduler.step()
        improved = False
        if validation is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            record.update(evaluate(model, validation, device, lpips_net, args.alpha))
            improved = record['psnr'] > best
            best = max(record['psnr'], best)
        history.append(record)
        state = dict(variant=variant, epoch=epoch, config=config, history=history, best_psnr=best, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        if improved:
            save_checkpoint(state, out / 'best.pt')
        save_checkpoint(state, latest)
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
