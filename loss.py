from torch import nn
import torch
from config import cfg


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, decouple_loss):
        differ = truth - pred  # s b c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # s b
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # s b
        mse = torch.mean(mse)  # 1
        mae = torch.mean(mae)  # 1
        loss = mse + mae
        if 'PredRNN-V2' in cfg.model_name:
            decouple_loss = torch.sum(decouple_loss, (1, 3))  # s l b c -> s b
            decouple_loss = torch.mean(decouple_loss)  # 1
            loss = loss + cfg.decouple_loss_weight * decouple_loss
        return loss
