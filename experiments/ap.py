"""Preserve M's direct dictionary modulation and Y's routing on top of W."""
import torch
from torch import nn

from experiments.m import RestorationDictionary
from experiments.runner import main, standard_backward
from experiments.w import RegionGateCIDNet


class DictionaryGateCIDNet(RegionGateCIDNet):
    def __init__(self):
        super().__init__()
        self.dictionary = RestorationDictionary()
        self.affine = nn.Linear(32, 144)
        self.dictionary_gain = nn.Parameter(torch.zeros(()))
        self.film = nn.Linear(32, 16)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.direct = True
        self.scene_router = True

    def routing_context(self, hvi):
        return super().routing_context(hvi), self.dictionary(hvi)

    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        regions, scene = context
        if self.scene_router:
            scale, shift = self.film(scene)[:, :, None, None].chunk(2, dim=1)
            regions = regions * (1 + scale) + shift
        features = self.HV_LCA2(hv, intensity)
        updated = self.mix_experts(features, self.router[2](regions))
        if self.direct:
            scale, shift = self.affine(scene)[:, :, None, None].chunk(2, dim=1)
            updated = updated + self.dictionary_gain * (features * scale.sigmoid() + shift)
        return updated

    def decode_intensity_with_context(self, intensity, attended, context):
        return super().decode_intensity_with_context(intensity, attended, context[0])


def options(parser):
    parser.add_argument('--no-direct', action='store_true')
    parser.add_argument('--no-scene-router', action='store_true')


def configure(model, config):
    model.direct = not config['no_direct']
    model.scene_router = not config['no_scene_router']


if __name__ == '__main__':
    main('ap', DictionaryGateCIDNet, standard_backward, options, configure=configure,
         resume_keys=('no_direct', 'no_scene_router'))
