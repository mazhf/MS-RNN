from torch.utils.data import Dataset
import os
import cv2
import sys
sys.path.append("..")
from config import cfg
import numpy as np
from util.meteo import dBZ2Pixel


def load_rain_f_data():
    train_data = Data(mode='train')
    test_data = Data(mode='test')
    return train_data, test_data, test_data


class Data(Dataset):
    def __init__(self, mode=''):
        super().__init__()
        years = [2017, 2018, 2019]
        all_rad_lis = []
        for year in years:
            year_pth = os.path.join(cfg.GLOBAL.DATASET_PATH, 'radar_' + str(year))
            rad_lis = os.listdir(year_pth)
            all_rad_lis = all_rad_lis + rad_lis
        all_rad_lis.sort(key=lambda x: int(x.split('-')[0]))
        split = int(6 * len(all_rad_lis) / 7)
        if mode == 'train':
            self.rad_lis = all_rad_lis[:split]
        else:
            self.rad_lis = all_rad_lis[split:]
        self.rad_num = len(self.rad_lis)
        self.LEN = cfg.in_len + cfg.out_len

    def __getitem__(self, idx):
        seq_pth_lis = self.rad_lis[idx * self.LEN: (idx + 1) * self.LEN]
        sequence = []
        for i in range(len(seq_pth_lis)):
            rad_name = seq_pth_lis[i]
            rad_par_dir = 'radar_' + rad_name[:4]
            rad_pth = os.path.join(cfg.GLOBAL.DATASET_PATH, rad_par_dir, rad_name)
            rad = np.load(rad_pth)
            rad = np.nan_to_num(rad)  # surface 'Rain' have nan pixel, others unsure
            rad = dBZ2Pixel(rad) * 255  # raw data maybe dBZ, which range from 0 to 99.83.
            rad = cv2.resize(rad, (cfg.width, cfg.height))
            rad = rad.astype(cfg.data_type)
            sequence.append(rad)
        sequence = np.array(sequence)  # S*H*W
        sequence = np.expand_dims(sequence, 1)  # S*1*H*W
        return sequence

    def __len__(self):
        return self.rad_num // self.LEN
