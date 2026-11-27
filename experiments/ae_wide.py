from experiments.ae import PhaseCIDNet, run


class WidePhaseCIDNet(PhaseCIDNet):
    widths = (64, 128)


if __name__ == '__main__':
    run('ae_wide', WidePhaseCIDNet)
