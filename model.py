from torch import nn
import torch
from config import cfg
import numpy as np
from util.utils import make_layers


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


class Model(nn.Module):
    def __init__(self, embed, rnn, fc):
        super().__init__()
        self.embed = make_layers(embed)
        self.rnns = rnn
        self.fc = make_layers(fc)
        self.in_len = cfg.in_len
        self.out_len = cfg.out_len
        self.eval_len = cfg.eval_len
        self.batch_size = cfg.batch
        self.use_ss = cfg.use_scheduled_sampling

    def forward(self, inputs, mode=''):
        x, eta = inputs  # sbchw
        if 'kth' in cfg.dataset:
            if mode == 'train':
                out_len = self.out_len
            else:
                out_len = self.eval_len
        else:
            out_len = self.out_len
        shape = [out_len] + list(x.shape)[1:]
        if self.use_ss:
            mask = scheduled_sampling(shape, eta)
        outputs = []
        layer_hiddens = None
        output = None
        m = None
        for t in range(x.shape[0] - 1):
            if t < self.in_len:
                input = x[t]
            else:
                if self.use_ss:
                    input = mask[t - self.in_len] * x[t] + (1 - mask[t - self.in_len]) * output
                else:
                    input = output
            output, m, layer_hiddens = self.rnns(input, m, layer_hiddens, self.embed, self.fc)
            outputs.append(output)
        outputs = torch.stack(outputs)  # sbchw
        return outputs





