import os.path
import logging
from copy import copy
import h5py
import pandas as pd
from datetime import datetime
import time
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from config import cfg
import cv2
import os
import numpy as np


logger = logging.getLogger(__name__)


def string2timestamp(strings, T=48):
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))
    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=20):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1)]
        i = len_closeness
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [np.transpose(self.get_matrix(self.pd_timestamps[i] - j * offset_frame), [1, 2, 0]) for j in
                   depends[0]]
            if len_closeness > 0:
                XC.append(np.stack(x_c, axis=0))
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.stack(XC, axis=0)
        return XC, timestamps_Y


def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname, 'r') as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)


class MinMaxNormalization(object):
    """
       MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        # X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def make_dataset(data_path, len_test, nb_flow=2, len_closeness=None, T=48):
    assert (len_closeness > 0)  # total input and output length
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(data_path, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    XC = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _timestamps_Y = st.create_dataset(len_closeness=len_closeness)
        XC.append(_XC)
        timestamps_Y += _timestamps_Y
    XC = np.concatenate(XC, axis=0)
    print("XC shape: ", XC.shape)
    XC_train = XC[:-len_test]
    XC_test = XC[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = XC_train
    X_test = XC_test
    print('train shape:', XC_train.shape, 'test shape: ', XC_test.shape)
    return X_train, X_test, mmn, timestamp_train, timestamp_test


class Data(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data  # b s h w

    def __getitem__(self, index):
        sample = self.data[index, ...]  # s h w
        sample = np.expand_dims(sample, 1)
        return sample  # S*C*H*W

    def __len__(self):
        return self.data.shape[0]


def load_taxiBJ():
    train, test, _, _, _ = make_dataset(data_path=cfg.GLOBAL.DATASET_PATH, len_test=48 * 7 * 4, nb_flow=2,
                                        len_closeness=cfg.in_len + cfg.out_len)
    train_data = Data((train[:, :, :, :, 0] * 255).astype(np.uint8))  # 只取一个通道，归一化后乘以255（有像素值为655的...）
    test_data = Data((test[:, :, :, :, 0] * 255).astype(np.uint8))
    valid_data = Data((test[:, :, :, :, 0] * 255).astype(np.uint8))

    return train_data, valid_data, test_data


if __name__ == '__main__':
    pth = '/home/mazhf/Spatiotemporal/dataset/taxiBJ'
    total_len = 8
    X_train, X_test, mmn, _, _ = make_dataset(data_path=pth, T=48, nb_flow=2, len_closeness=total_len,
                                              len_test=48 * 7 * 4)
    # test
    test_pth = '/home/mazhf/Spatiotemporal/dataset/taxiBJ_test'
    if not os.path.exists(test_pth):
        os.makedirs(test_pth)

    # X_test_raw = mmn.inverse_transform(X_test)
    X_test_raw = X_test[:, :, :, :, 0]
    for i in range(100):
        for j in range(X_test_raw.shape[1]):
            save_pth = os.path.join(test_pth, str(i) + '_' + str(j) + '.png')
            # c1, c2 = cv2.split(X_test_raw[i, j, ...])
            # c3 = np.zeros_like(c1)
            # # c3 = np.ones_like(c1)
            # c_all = cv2.merge([c3, c2, c1]) * 255  # BGR
            # c_all = c_all.astype(np.uint8)
            # cv2.imwrite(save_pth, c_all)
            cv2.imwrite(save_pth, X_test_raw[i, j, ...])
