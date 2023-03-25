import os
from config import cfg

# gpus 需要放在torch之前
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
from loss import Loss
from train_and_test import train_and_test
from net_params import nets
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.load_data import load_data
import argparse


# fix init
def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 固定random.random()生成的随机数
    np.random.seed(seed)  # 固定np.random()生成的随机数
    torch.manual_seed(seed)  # 固定CPU生成的随机数
    torch.cuda.manual_seed(seed)  # 固定GPU生成的随机数-单卡
    torch.cuda.manual_seed_all(seed)  # 固定GPU生成的随机数-多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


fix_random(2022)

# params
gpu_nums = cfg.gpu_nums
batch_size = cfg.batch
train_epoch = cfg.epoch
valid_epoch = cfg.valid_epoch
LR = cfg.LR

# model
model = Model(nets[0], nets[1], nets[2])

# run: python -m torch.distributed.launch --nproc_per_node=4 --master_port 39985 main.py
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
print('local_rank: ', args.local_rank)

# parallel group
torch.distributed.init_process_group(backend="nccl")

# model parallel
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank],
                                            output_device=args.local_rank)

threads = cfg.dataloader_thread
train_data, valid_data, test_data = load_data()
train_sampler = DistributedSampler(train_data, shuffle=True)
valid_sampler = DistributedSampler(valid_data, shuffle=False)
train_loader = DataLoader(train_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=train_sampler)
test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=False)
valid_loader = DataLoader(valid_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=valid_sampler)
loader = [train_loader, test_loader, valid_loader]

# optimizer
if cfg.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = None

# loss
criterion = Loss().cuda()

# train valid test
train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler)
