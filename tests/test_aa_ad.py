"""Numerical checks for exposure coordinates, splatting, color slicing and held-out pixels."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiments.aa import ExposurePairs, sample_exposure
from experiments.ab import gaussian_upsample
from experiments.ac import BilateralColorCIDNet
from experiments.ad import next_scale_pair


class RestorationModulesTest(unittest.TestCase):
    def test_flipped_crop_uses_matching_full_frame_coordinates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
            image = Image.fromarray(pixels).convert('RGB')
            for folder in ('low', 'high'):
                (root / folder).mkdir()
                image.save(root / folder / '0.png')
            pairs = ExposurePairs(SimpleNamespace(train_dir=root, crop=24, context_mode='aligned'))
            with patch('experiments.aa.random.randint', side_effect=[7, 20]), \
                    patch('experiments.aa.random.random', side_effect=[0., 0.]):
                crop, small, stats, box, target = pairs[0]
            sampled = sample_exposure(small[None], box[None], (24, 24))[0]
            torch.testing.assert_close(sampled, crop)
            sampled_gt = sample_exposure(stats[None], box[None], (24, 24))[0]
            torch.testing.assert_close(sampled_gt, target[:2])

    def test_splat_matches_direct_accumulation_including_borders(self):
        values = torch.arange(6, dtype=torch.float64).view(1, 1, 2, 3)
        geometry = torch.zeros(1, 5, 2, 3, dtype=values.dtype, requires_grad=True)
        actual = gaussian_upsample(values, geometry, 'offset')
        y, x = torch.meshgrid(torch.arange(4), torch.arange(6), indexing='ij')
        weighted, mass = torch.zeros_like(actual), torch.zeros_like(actual)
        for i in range(2):
            for j in range(3):
                dx, dy = x.double() - (2 * j + 0.5), y.double() - (2 * i + 0.5)
                weight = torch.exp(-0.5 * (dx.square() + dy.square()) / 1.25 ** 2)
                weight *= (dx.abs() <= 1.5) & (dy.abs() <= 1.5)
                weighted += weight * values[0, 0, i, j]
                mass += weight
        torch.testing.assert_close(actual, weighted / mass)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertGreater(geometry.grad[:, :2].abs().sum().item(), 0.)

    def test_bilateral_identity_and_guide_axis(self):
        model = BilateralColorCIDNet()
        hvi = torch.rand(1, 3, 16, 24)
        features = torch.randn(1, 72, 4, 6)
        torch.testing.assert_close(model.correct_chroma(hvi, features), hvi, rtol=0, atol=0)
        with torch.no_grad():
            model.coefficients.bias.view(7, 8)[4].copy_(torch.arange(8.))
        # At the eight depth-bin centers the sampled H bias is exactly the bin index.
        bins = torch.arange(8.).repeat(3)
        hvi[:, 2] = (bins + 0.5) / 8
        delta = model.correct_chroma(hvi, features) - hvi
        torch.testing.assert_close(delta[:, 0], bins.expand_as(delta[:, 0]))
        torch.testing.assert_close(delta[:, 1:], torch.zeros_like(delta[:, 1:]))

    def test_nsp_input_and_four_targets_use_distinct_source_pixels(self):
        # Encode absolute source position, including incomplete border tiles.
        image = torch.arange(17 * 23.).view(1, 1, 17, 23).expand(2, 3, -1, -1)
        sample, target = next_scale_pair(image)
        patches = target.unfold(2, 2, 2).unfold(3, 2, 2)
        self.assertTrue((patches != sample[..., None, None]).all())
        corner = patches[..., 0, 0]
        torch.testing.assert_close(patches[..., 0, 1], corner + 1)
        torch.testing.assert_close(patches[..., 1, 0], corner + 23)
        torch.testing.assert_close(patches[..., 1, 1], corner + 24)
        self.assertTrue(((corner // 23 // 5) == (sample // 23 // 5)).all())
        self.assertTrue(((corner % 23 // 5) == (sample % 23 // 5)).all())
