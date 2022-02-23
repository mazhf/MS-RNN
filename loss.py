from torch import nn
from config import cfg
import torch
from util.utils import rainfall_to_pixel
from util.numba_accelerated import compute_focal_numba


class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS=cfg.normal_mae_mse_loss):
        super().__init__()
        self.pixel_weights_lis = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        self.NORMAL_LOSS = NORMAL_LOSS
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.all_len = cfg.in_len + cfg.out_len - 2
        self.all_epoch = cfg.epoch

    def forward(self, truth, pred, epoch):
        differ = truth - pred  # s b c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # s b
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # s b
        mse = self.mse_weight * torch.mean(mse)
        mae = self.mae_weight * torch.mean(mae)
        loss = self.NORMAL_LOSS * (mse + mae)
        return loss

