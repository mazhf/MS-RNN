from torch import nn
import torch
from models.predrnn_v2 import PredRNN_V2_Cell
import torch.nn.functional as F
import sys
sys.path.append("..")
from config import cfg


class MS_PredRNN_V2(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [PredRNN_V2_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                PredRNN_V2_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_V2_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_V2_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_V2_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_V2_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)
        self.adapter = cfg.LSTM_conv(input_channel, output_channel, 1, 1, 0)
        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups_m = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is MS-PredRNN-V2!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        out = []
        decouple_loss = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, m, next_hiddens, delta_c, delta_m = self.lstm[l](x, m, hiddens)
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
                m = self.downs_m[0](m)
            elif l == 1:
                x = self.downs[1](x)
                m = self.downs_m[1](m)
            elif l == 3:
                x = self.ups[0](x) + out[1]
                m = self.ups_m[0](m)
            elif l == 4:
                x = self.ups[1](x) + out[0]
                m = self.ups_m[1](m)
            delta_c = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)  # b c h*w
            delta_m = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)  # b c h*w
            decouple_loss.append(torch.abs(torch.cosine_similarity(delta_c, delta_m, dim=2)))  # b c
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        decouple_loss = torch.stack(decouple_loss)  # l b c
        return x, m, next_layer_hiddens, decouple_loss
