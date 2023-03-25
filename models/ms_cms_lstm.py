from torch import nn
import torch
from models.cms_lstm import CMS_LSTM_Cell
import sys
sys.path.append("..")
from config import cfg


class MS_CMS_LSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [CMS_LSTM_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                CMS_LSTM_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                CMS_LSTM_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                CMS_LSTM_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                CMS_LSTM_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                CMS_LSTM_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is MS-CMS-LSTM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        out = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, next_hiddens = self.lstm[l](x, hiddens)
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
            elif l == 1:
                x = self.downs[1](x)
            elif l == 3:
                x = self.ups[0](x) + out[1]
            elif l == 4:
                x = self.ups[1](x) + out[0]
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss