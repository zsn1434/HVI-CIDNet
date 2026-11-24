from net.CIDNet import CIDNet
from experiments.runner import main, standard_backward

if __name__ == '__main__':
    main('baseline', CIDNet, standard_backward)
