from experiments.af import ResidualCIDNet, run


class WideResidualCIDNet(ResidualCIDNet):
    coarse_width = 48


if __name__ == '__main__':
    run('af_wide', WideResidualCIDNet)
