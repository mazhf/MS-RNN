from torch import nn
import torch
from models.preciplstm import PrecipLSTM_cell
import sys
sys.path.append("..")
from config import cfg


class MS_PrecipLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [PrecipLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                PrecipLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PrecipLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PrecipLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PrecipLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PrecipLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)
        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups_m = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        print('This is MS-PrecipLSTM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        x_t_1_lis = []
        out = []
        for l in range(self.n_layers):
            x_t_1_lis.append(x)  # x h0 h1 ... hl-2, save inputs in t-1
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
                x_t_1 = layer_hiddens[-1][l]
            else:
                hiddens = None
                x_t_1 = None
            x, m, next_hiddens = self.lstm[l](x, x_t_1, m, hiddens)
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
            next_layer_hiddens.append(next_hiddens)
        next_layer_hiddens.append(x_t_1_lis)  # l+1
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss
