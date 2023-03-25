from torch.utils.data import Dataset
import sys
sys.path.append("..")
from config import cfg
import numpy as np
import cv2
import os


def load_moving_mnist():
    train_data = Data(mode='train')
    test_data = Data(mode='test')
    return train_data, test_data, test_data


def read_moving_data(path, mode):
    # mnist数据集本身是随机的.直接用切片划分np数组,按照7:3划分数据集
    data_all = np.load(path + '.npy')
    s, b, h, w = data_all.shape
    if mode == 'train':
        data = data_all[:, :b * 7 // 10, :, :]
    else:
        data = data_all[:, b * 7 // 10:, :, :]
    return data


class Data(Dataset):
    def __init__(self, mode=''):
        super().__init__()
        _base_path = cfg.GLOBAL.DATASET_PATH
        self.mode = mode
        self.data = read_moving_data(_base_path, self.mode)

    def __getitem__(self, index):
        sample = self.data[:, index, :, :]
        sample = np.expand_dims(sample, 1)
        return sample  # S*C*H*W

    def __len__(self):
        return self.data.shape[1]


def test():
    data = np.load('/home/mazhf/Spatiotemporal/moving-mnist-30-test.npy')
    demo = data[:, :100, ...]
    for i in range(demo.shape[1]):
        seq = demo[:, i, ...]
        a = np.zeros((64, 64))
        for j in range(seq.shape[0]):
            a = np.concatenate((a, seq[j]), axis=1)
        cv2.imwrite(os.path.join('/home/mazhf/Spatiotemporal/test', str(i) + '.png'), a)
        print(i)


if __name__ == '__main__':
    test()
