"""Y: use M's scene dictionary to condition C's regional expert routing."""
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.c import RegionCIDNet
from experiments.m import RestorationDictionary
from experiments.runner import main, padded_input, standard_backward


class DictionaryRouterCIDNet(RegionCIDNet):
    def __init__(self):
        super().__init__()
        self.prior = RestorationDictionary()
        self.film = nn.Linear(32, 16)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def routing_context(self, hvi):
        return self.prior(hvi)

    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        scale, shift = self.film(context)[:, :, None, None].chunk(2, dim=1)
        hidden = self.router_hidden(self.region_statistics(hvi))
        logits = self.router[2](hidden * (1 + scale) + shift)
        return self.mix_experts(self.HV_LCA2(hv, intensity), logits)


def options(parser):
    parser.add_argument('--descriptor', choices=('own', 'mean', 'other'), default='own',
                        help='Evaluation only: substitute descriptors across the evaluation set')


@torch.inference_mode()
def evaluation_predictor(model, pairs, device, args):
    if args.descriptor == 'own':
        return None
    if args.descriptor == 'other' and len(pairs) < 2:
        raise ValueError('--descriptor other needs at least two paired images')
    model.eval()
    descriptors = []
    for low, _ in DataLoader(pairs, batch_size=1):
        hvi = model.HVIT(padded_input(low.to(device)))
        descriptors.append(model.routing_context(hvi))
    descriptors = torch.cat(descriptors)
    if args.descriptor == 'mean':
        descriptors = descriptors.mean(0, keepdim=True).expand(len(pairs), -1)
    else:
        descriptors = descriptors.roll(-1, dims=0)

    def predict(rgb, index):
        return model.enhance(*model.input_features(rgb), context=descriptors[index:index + 1])

    return predict


def configure(model, config):
    if config.get('command') == 'train' and config.get('descriptor', 'own') != 'own':
        raise ValueError('--descriptor mean/other are eval-only controls')


if __name__ == '__main__':
    main('y', DictionaryRouterCIDNet, standard_backward, options, configure=configure,
         evaluation_predictor=evaluation_predictor)
