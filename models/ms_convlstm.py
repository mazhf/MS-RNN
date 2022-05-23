from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class ConvLSTM_cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens
        x2h = self._conv_x2h(x)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        o = torch.sigmoid(o)
        next_h = o * torch.tanh(next_c)

        ouput = next_h
        next_hiddens = [next_h, next_c]
        return ouput, next_hiddens


class Multi_Scale_ConvLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is Multi Scale ConvLSTM!')

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
        return x, m, next_layer_hiddens
