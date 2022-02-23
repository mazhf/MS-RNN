import os
from config import cfg

# gpus 需要放在torch之前
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
from torch.optim import lr_scheduler
from loss import Weighted_mse_mae
from train_and_test import train_and_test
from net_params import params
import random
import numpy as np
from hypergrad import SGDHD, AdamHD
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


fix_random(2021)

# params
gpu_nums = cfg.gpu_nums
batch_size = cfg.batch
train_epoch = cfg.epoch
valid_epoch = cfg.valid_epoch
save_checkpoint_epoch = cfg.valid_epoch
LR = cfg.LR

# model
model = Model(params[0], params[1], params[2])

# optimizer
if cfg.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # lr_schedul = lr_scheduler.StepLR(optimizer, step_size=LR_epoch_size, gamma=gamma)
elif cfg.optimizer == 'SGDHD':
    optimizer = SGDHD(model.parameters(), lr=LR, hypergrad_lr=1e-8)
elif cfg.optimizer == 'AdamHD':
    optimizer = AdamHD(model.parameters(), lr=LR, hypergrad_lr=1e-8)

# 设置并行——以下设置顺序不可颠倒！run: python -m torch.distributed.launch --nproc_per_node=4 --master_port 39985 main.py
# torch.distributed.launch arguments 该参数只能这么用，其他参数只能放在命令行。。。
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

# dataloader DataLoader的shuffle和DistributedSampler的shuffle为True只能使用一个，valid和test可以shuffle，但是为了取test demo就不了
threads = cfg.dataloader_thread
train_data, valid_data, test_data = load_data()
train_sampler = DistributedSampler(train_data, shuffle=True)
valid_sampler = DistributedSampler(valid_data, shuffle=False)
train_loader = DataLoader(train_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=train_sampler)
# 通常情况下，数据在内存中要么以锁页的方式存在，要么保存在虚拟内存(磁盘)中，设置为True后，数据直接保存在锁页内存中，后续直接传入cuda；
# 否则需要先从虚拟内存中传入锁页内存中，再传入cuda，这样就比较耗时了，但是对于内存的大小要求比较高。
# 先把dataset读到CPU上，然后GPU只读每个batch的数据，ucf50数据集太大，导致训练集和验证集都加载完后，没有test的内存地方了
test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=False)
valid_loader = DataLoader(valid_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=valid_sampler)
loader = [train_loader, test_loader, valid_loader]

# loss
criterion = Weighted_mse_mae().cuda()

# train valid test
train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, save_checkpoint_epoch, loader, train_sampler)
