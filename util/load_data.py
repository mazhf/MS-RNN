import sys
sys.path.append("..")
from config import cfg
from util.load_mnist import load_moving_mnist
from util.load_kth import KTH
from util.load_kth_only_run import KTH_Only
from util.load_taxibj import load_taxiBJ
from util.load_human import load_hmn
from util.load_hko import load_HKO
from util.load_rain_f import load_rain_f_data


def load_data():
    if 'mnist' in cfg.dataset:
        return load_moving_mnist()
    elif 'kth' in cfg.dataset:
        if cfg.kth_only_run:
            kth_ = KTH_Only
        else:
            kth_ = KTH
        train_data = kth_(train=True, seq_len=cfg.in_len + cfg.out_len)
        test_data = kth_(train=False, seq_len=cfg.in_len + cfg.eval_len)
        valid_data = test_data
        return train_data, valid_data, test_data
    elif 'taxiBJ' in cfg.dataset:
        return load_taxiBJ()
    elif 'human3.6m' in cfg.dataset:
        return load_hmn()
    elif 'HKO-7' in cfg.dataset:
        return load_HKO()
    elif 'DWD' in cfg.dataset:
        return load_HKO()
    elif 'MeteoNet' in cfg.dataset:
        return load_HKO()
    elif 'RAIN-F' in cfg.dataset:
        return load_rain_f_data()




