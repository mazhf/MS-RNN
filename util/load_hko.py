from torch.utils.data import Dataset
import os
import cv2
import sys
sys.path.append("..")
from config import cfg
import numpy as np


def load_HKO():
    train_data = Data(mode='train')
    test_data = Data(mode='test')
    valid_data = Data(mode='valid')
    return train_data, valid_data, test_data


class Data(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        _base_path = cfg.GLOBAL.DATASET_PATH
        self._data_path = os.path.join(_base_path, mode)
        self._img_path_list = os.listdir(self._data_path)
        self._img_path_list.sort(key=lambda x: int(x.split('.')[0]))
        self.img_num = len(self._img_path_list)
        self.IN_LEN = cfg.in_len
        self.OUT_LEN = cfg.out_len
        self.LEN = self.IN_LEN + self.OUT_LEN
        self.mode = mode

    def __getitem__(self, index):
        # train,test，valid都为按次序取，最后用dataloder的shuffle处理
        img_1_batch_path_list = self._img_path_list[index * self.LEN: (index + 1) * self.LEN]
        l = []
        for i in range(len(img_1_batch_path_list)):
            img_path = img_1_batch_path_list[i]
            img = cv2.imread(os.path.join(self._data_path, img_path), flags=0)  # H*W
            img = cv2.resize(img, (cfg.width, cfg.height))
            # 默认参数为1，读取RGB图像，因此为三通道（3维数组），flags=0时，是单通道（2维数组）
            l.append(img)
        l = np.array(l)  # S*H*W
        frames = np.expand_dims(l, 1)  # S*1*H*W

        return frames  # S*C*H*W

    def __len__(self):
        return self.img_num // self.LEN