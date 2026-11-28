"""Joint AP and AQ with one backbone, dictionary, and regional expert bank."""
from experiments import ap, aq
from experiments.r import backward
from experiments.runner import main


class JointCIDNet(ap.DictionaryGateCIDNet, aq.ScaleGuidedBands):
    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        routing, scale = context
        features = ap.DictionaryGateCIDNet.encode_hv_skip(self, hv, intensity, hvi, routing, auxiliary)
        return self.apply_information(features, hvi, scale)

    def decode_intensity_with_context(self, intensity, attended, context):
        routing, scale = context
        w_update = ap.DictionaryGateCIDNet.decode_intensity_with_context(self, intensity, attended, routing)
        return w_update + self.band_update(intensity, attended - intensity, scale)


def options(parser):
    ap.options(parser)
    aq.options(parser)


def configure(model, config):
    ap.configure(model, config)
    aq.configure(model, config)


if __name__ == '__main__':
    main('as_fusion', JointCIDNet, backward, options, configure=configure,
         resume_keys=('no_direct', 'no_scene_router', 'detail_mode'))
