import numpy as np
import h5py
import ast
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("..")
from config import cfg
import cv2

"""太慢了，重写后和load_HKO合并为load_data"""


def R_to_dBZ(R):
    Z = 256 * R ** 1.42
    dBZ = 10 * np.log10(Z)
    return dBZ


def dBZ_to_R(dBZ):
    Z = 10 ** (dBZ / 10)
    R = (Z / 256) ** (1 / 1.42)
    return R

print(dBZ_to_R(30))

def dBZ_to_P(dBZ):
    # pixel value: 0 - 1
    P = np.clip((dBZ + 10.0) / 70.0, a_min=0.0, a_max=1.0)
    return P


def P_to_dBZ(P):
    dBZ = P * 70.0 - 10.0
    return dBZ


def P_to_R(P):
    dBZ = P_to_dBZ(P)
    R = dBZ_to_R(dBZ)
    return R


def R_to_P(R):
    dBZ = R_to_dBZ(R)
    P = dBZ_to_P(dBZ)
    return P


def resize_img(x):
    h = cfg.height
    w = cfg.width
    x = np.transpose(x, (1, 2, 0))  # H*W*S
    # x = cv2.resize(x, (w, h))  # h*w*S
    x = np.transpose(x, (2, 0, 1))  # S*h*w
    return x


def data_preprocessing(X):
    X = np.array(X)
    X = X * 12  # mm/5min ——> mm/h
    X = R_to_P(X)  # 900x900
    X = resize_img(X)  # resize
    return X


class DataGenerator(Dataset):
    def __init__(self, dataset_dict, image_names):
        self.keys = image_names
        self.data = dataset_dict
        self.IN_LEN = cfg.in_len
        self.OUT_LEN = cfg.out_len
        self.LEN = self.IN_LEN + self.OUT_LEN

    def get_index(self, idx):
        x = []
        for i in range(self.LEN):
            try:
                arr = self.data.get(self.keys[idx * self.LEN + i])  # H W
            except:
                print(idx * self.LEN + i)
            x.append(arr)
        x = data_preprocessing(x)  # S H W
        return x

    def __getitem__(self, index):
        x = self.get_index(index)  # S H W
        x = x[:, np.newaxis, ...]  # S 1 H W
        return x

    def __len__(self):
        return len(self.keys) // self.LEN


def load_dwd():
    dataset_dict = h5py.File('/home/mazhf/Precipitation-Nowcasting/DWD-12/RYDL.hdf5', 'r')
    with open('/home/mazhf/Precipitation-Nowcasting/DWD-12/RYDL_keys.txt', 'r') as f:
        image_names = ast.literal_eval(f.read())
    image_names = [name for name in image_names]
    train_images = [name for name in image_names if name[0:4] in ['2006', '2007', '2008', '2009', '2010', '2011',
                                                                  '2012', '2013', '2014']]
    val_images = [name for name in image_names if name[0:4] == '2015']
    test_images = [name for name in image_names if name[0:4] in ['2016', '2017']]
    train_gen = DataGenerator(dataset_dict=dataset_dict, image_names=train_images)
    val_gen = DataGenerator(dataset_dict=dataset_dict, image_names=val_images)
    test_gen = DataGenerator(dataset_dict=dataset_dict, image_names=test_images)

    return train_gen, val_gen, test_gen


def test():
    import cv2
    import os
    train_gen, _, _ = load_dwd()
    train_loader = DataLoader(train_gen, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
    c = 0
    save_path = r'/home/mazhf/Precipitation-Nowcasting/test'
    if not os.path.exists(save_path):
        print("# path not exists")
        os.makedirs(save_path)
    for i, train_batch in enumerate(train_loader):
        if c < 500:
            for j in range(train_batch.shape[1]):
                img = train_batch[:, j, ...]  # B 1 H W
                img = img.squeeze(dim=0)  # C H W
                img = (img.numpy() * 255).astype(np.uint8)  # 0-255
                cv2.imwrite(os.path.join(save_path, str(c) + '.png'), img[0])
                c += 1
                print(c)
        else:
            break
    print('done!')


if __name__ == '__main__':
    test()
