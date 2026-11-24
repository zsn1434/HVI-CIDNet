"""Train CIDNet with an offline TM-BSN denoising target."""
import argparse
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from experiments.runner import Pairs, image_index, load_model, main, save_checkpoint
from net.CIDNet import CIDNet


class DenoisingPairs(Pairs):
    def __init__(self, args):
        super().__init__(args.train_dir, args.crop)
        if args.teacher_dir is None:
            if not args.identity_teacher:
                raise ValueError("V requires --teacher-dir or --identity-teacher")
            self.samples = [(low, low, target) for low, target in self.pairs]
            return
        images = image_index(args.teacher_dir)
        low_folder = self.pairs[0][0].parent
        low_images = image_index(low_folder)
        if images.keys() != low_images.keys():
            raise ValueError(f"Teacher targets in {args.teacher_dir} do not match {low_folder}")
        key_by_path = {path: key for key, path in low_images.items()}
        self.samples = [(low, images[key_by_path[low]], target) for low, target in self.pairs]

    def __getitem__(self, index):
        return self.load_images(self.samples[index])


class DistillationCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.denoising_head = nn.Conv2d(36, 3, 1)

    def with_denoising(self, low):
        hvi, stem = self.input_features(low)
        return self.enhance(hvi, stem), self.denoising_head(stem)


def options(parser):
    teacher = parser.add_mutually_exclusive_group()
    teacher.add_argument("--teacher-dir", type=Path, help="Prepared low-light RGB targets")
    teacher.add_argument("--identity-teacher", action="store_true", help="Use input as auxiliary target")
    parser.add_argument("--teacher-weight", type=float, default=0.1)


def backward(model, low, denoised, target, objective, args):
    output, denoising = model.with_denoising(low)
    reconstruction, perceptual = objective(model, output, target)
    loss = reconstruction + perceptual + args.teacher_weight * F.l1_loss(denoising, denoised)
    loss.backward()
    return loss.detach()


@torch.inference_mode()
def denoise_image(model, x, tile_size, halo, hole_size):
    height, width = x.shape[-2:]
    restored = torch.empty_like(x)
    for y in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            bottom, right = min(y + tile_size, height), min(col + tile_size, width)
            top_context, left_context = max(y - halo, 0), max(col - halo, 0)
            tile = x[..., top_context:bottom + halo, left_context:right + halo]
            h, w = tile.shape[-2:]
            # Upstream concatenates four rotations along the batch dimension.
            side = max(h, w)
            tile = F.pad(tile, (0, side - w, 0, side - h), mode="replicate")
            pred = model(tile, hole_size=hole_size)
            restored[..., y:bottom, col:right] = pred[..., y - top_context:bottom - top_context,
                                                       col - left_context:right - left_context]
    return restored


def prepare():
    parser = argparse.ArgumentParser(description="Generate same-exposure targets with the official TM-BSN teacher")
    parser.add_argument("--teacher-source", type=Path, required=True, help="Path to upstream model/TMBSN.py")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--halo", type=int, default=64)
    parser.add_argument("--hole-size", type=int, default=4)
    args = parser.parse_args(sys.argv[2:])
    if args.tile < 1 or args.halo < 0 or not 1 <= args.hole_size <= args.tile:
        parser.error("tile must be positive, halo nonnegative, and hole-size in [1, tile]")
    images = image_index(args.low_dir)
    if not images:
        parser.error(f"No images in {args.low_dir}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Teacher output directory must be empty")

    spec = importlib.util.spec_from_file_location("upstream_tmbsn", args.teacher_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TMBSN()
    model.load_state_dict(torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True))
    model.to(args.device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teacher-", dir=args.out.parent) as work:
        staging = Path(work) / "images"
        staging.mkdir()
        for path in images.values():
            with Image.open(path) as image:
                low = to_tensor(image.convert("RGB"))[None].to(args.device)
            target = denoise_image(model, low, args.tile, args.halo, args.hole_size)
            to_pil_image(target[0].clamp(0, 1).cpu()).save(staging / f"{path.stem}.png")
            print(path.name, flush=True)
        staging.rename(args.out)


def export():
    parser = argparse.ArgumentParser(description="Export V as a plain CIDNet state_dict")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    if args.out.exists() or args.out.suffix == ".tmp":
        parser.error("Use a new output filename with a suffix other than .tmp")
    model = DistillationCIDNet()
    load_model(args.checkpoint, model, "v")
    del model.denoising_head
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.state_dict(), args.out)


def training_pairs(args):
    if not math.isfinite(args.teacher_weight) or args.teacher_weight < 0:
        raise ValueError("--teacher-weight must be finite and nonnegative")
    return DenoisingPairs(args)


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare"]:
        prepare()
    elif sys.argv[1:2] == ["export"]:
        export()
    else:
        main("v", DistillationCIDNet, backward, options, train_dataset=training_pairs,
             resume_keys=("teacher_dir", "identity_teacher", "teacher_weight"))
