import random
import os
import numpy as np
import torch
import cv2
import sys
sys.path.append("..")
from config import cfg


def load_file(data_root, data_type, seq_len, T):
    if data_type == 'test':
        file_path = os.path.join(os.path.join(os.path.dirname(data_root), 'kth_txt'), 'test_data_list_trimmed.txt')
    else:
        file_path = os.path.join(os.path.join(os.path.dirname(data_root), 'kth_txt'), 'train_data_list_trimmed.txt')
    with open(file_path) as f:
        vid_list = f.readlines()
    seq_list = []
    clip_dict = {}
    specific = ['jogging', 'running']
    num = 0
    for vid in vid_list:
        vid = vid[:-1]
        comps = vid.split(' ')
        name = comps[0]
        i_s = int(comps[1])
        i_e = int(comps[2])
        [ID, c, d] = name.split('_')
        if data_type == 'train':
            num += (i_e - seq_len + 1 - i_s + 1)
            for i in range(i_s, i_e - seq_len + 2):
                istart = i
                iend = i + seq_len  # not -1, therefore, when read it, just range(istart, iend)
                seq_list.append({'class': c, 'name': name, 'start': istart, 'end': iend})
                if name not in clip_dict.keys():
                    clip_dict[name] = []
                clip_dict[name].append([istart, iend])
        else:
            if c in specific:
                n_skip = 3
            else:
                n_skip = T
            for i in range(i_s, i_e - seq_len + 2, n_skip):
                istart = i
                iend = i + seq_len  # not -1, therefore, when read it, just range(istart, iend)
                seq_list.append({'class': c, 'name': name, 'start': istart, 'end': iend})
                if name not in clip_dict.keys():
                    clip_dict[name] = []
                clip_dict[name].append([istart, iend])
    random.shuffle(seq_list)
    print('total seq %d' % len(seq_list))
    print('total video %d' % len(clip_dict))
    return seq_list, clip_dict


class KTH(object):
    def __init__(self, train, seq_len=20, T=20):
        self.data_root = cfg.GLOBAL.DATASET_PATH
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        if train:
            self.train = True
            data_type = 'train'
            self.persons = list(range(1, 17))
        else:
            self.train = False
            self.persons = list(range(17, 26))
            data_type = 'test'
        random.seed(123)
        self.seq_list, self.clip_dict = load_file(self.data_root, data_type, seq_len, T)
        self.seed_set = False

    def get_sequence(self, index):
        vid = self.seq_list[index]
        c = vid['class']
        name = vid['name']
        istart = vid['start']
        iend = vid['end']
        dname = '%s/%s/%s' % (self.data_root, c, name)
        seq = []
        for i in range(istart, iend):
            fname = '%s/%s' % (dname, '%d.png' % i)
            im = cv2.imread(fname)
            im = cv2.resize(im, (cfg.width, cfg.height))
            seq.append(im[:, :, 0])  # 读黑白的  # S H W
        return np.expand_dims(np.array(seq), 1)  # S*C*H*W

    def __getitem__(self, index):
        if self.train:
            return torch.from_numpy(self.get_sequence(index))  # main中的dataloader处理shuffle
        else:
            return torch.from_numpy(self.get_sequence(index))

    def __len__(self):
        return len(self.seq_list)


