from torch import nn
import torch
from config import cfg
import numpy as np
from util.utils import make_layers
import math


def scheduled_sampling(shape, eta):
    S, B, C, H, W = shape
    # 随机种子已固定, 生成[0,1)随机数，形状 = (pre_len-1行，batch_size列)
    random_flip = np.random.random_sample((S - 1, B))  # outS-1 * B
    true_token = (random_flip < eta)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
    one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
    zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
    masks = []
    for t in range(S - 1):
        masks_b = []  # B*C*H*W
        for i in range(B):
            if true_token[t, i]:
                masks_b.append(one)
            else:
                masks_b.append(zero)
        mask = torch.cat(masks_b, 0)  # along batch size
        masks.append(mask)  # outS-1 * B*C*H*W
    return masks


def reverse_scheduled_sampling(shape_r, epoch):
    start_epoch = cfg.epoch / 3
    end_epoch = cfg.epoch * 2 / 3
    step = start_epoch / 5
    if epoch < start_epoch:
        eta_r = 0.5
    elif epoch < end_epoch:
        eta_r = 1.0 - 0.5 * math.exp(-float(epoch - start_epoch) / step)
    else:
        eta_r = 1.0
    if epoch < start_epoch:
        eta = 0.5
    elif epoch < end_epoch:
        eta = 0.5 - 0.5 * (epoch - start_epoch) / (end_epoch - start_epoch)
    else:
        eta = 0.0
    S, B, C, H, W = shape_r
    random_flip_r = np.random.random_sample((cfg.in_len - 1, B))  # inS-1 * B
    random_flip = np.random.random_sample((S - cfg.in_len - 1, B))  # outS-1 * B
    true_token_r = (random_flip_r < eta_r)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
    true_token = (random_flip < eta)  # 若eta为0，true_token[t, i]全部为False，mask元素全为0
    one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
    zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
    masks = []
    for t in range(S - 2):
        if t < cfg.in_len - 1:
            masks_b = []  # B*C*H*W
            for i in range(B):
                if true_token_r[t, i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batch size
            masks.append(mask)  # inS-1 * B*C*H*W
        else:
            masks_b = []  # B*C*H*W
            for i in range(B):
                if true_token[t - (cfg.in_len - 1), i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batch size
            masks.append(mask)  # outS-1 * B*C*H*W
    return masks


class Model(nn.Module):
    def __init__(self, embed, rnn, fc):
        super().__init__()
        self.embed = make_layers(embed)
        self.rnns = rnn
        self.fc = make_layers(fc)
        self.use_ss = cfg.scheduled_sampling
        self.use_rss = cfg.reverse_scheduled_sampling

    def forward(self, inputs, mode=''):
        x, eta, epoch = inputs  # s b c h w
        if 'kth' in cfg.dataset:
            if mode == 'train':
                out_len = cfg.out_len
            else:
                out_len = cfg.eval_len
        else:
            out_len = cfg.out_len
        shape = [out_len] + list(x.shape)[1:]
        shape_r = [cfg.in_len + out_len] + list(x.shape)[1:]
        if self.use_rss:
            mask = reverse_scheduled_sampling(shape_r, epoch)
        elif self.use_ss:
            mask = scheduled_sampling(shape, eta)
        outputs = []
        decouple_losses = []
        layer_hiddens = None
        output = None
        m = None
        for t in range(x.shape[0] - 1):
            if self.use_rss:
                if t == 0:
                    input = x[t]
                else:
                    input = mask[t - 1] * x[t] + (1 - mask[t - 1]) * output
            else:
                if t < cfg.in_len:
                    input = x[t]
                else:
                    if self.use_ss:
                        input = mask[t - cfg.in_len] * x[t] + (1 - mask[t - cfg.in_len]) * output
                    else:
                        input = output
            output, m, layer_hiddens, decouple_loss = self.rnns(input, m, layer_hiddens, self.embed, self.fc)
            outputs.append(output)
            decouple_losses.append(decouple_loss)
        outputs = torch.stack(outputs)  # s b c h w
        decouple_losses = torch.stack(decouple_losses)  # s l b c
        return outputs, decouple_losses





