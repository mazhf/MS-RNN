from torch.utils.data import Dataset
import sys
sys.path.append("..")
from util.human_preproccess import get_data


def load_hmn():
    train_data = Data(mode='train')
    test_data = Data(mode='test')
    valid_data = test_data
    return train_data, valid_data, test_data


class Data(Dataset):
    def __init__(self, mode=''):
        super().__init__()
        self.mode = mode
        train_data, test_data = get_data()
        if mode == 'train':
            self.data = train_data
        else:
            self.data = test_data

    def __getitem__(self, index):
        return self.data[index]  # S*C*H*W

    def __len__(self):
        return self.data.shape[0]
