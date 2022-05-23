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


class Multi_Scale_Fully_Connected_ConvLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstms = [ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                 ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                 ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                 ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                 ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                 ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstms = nn.ModuleList(lstms)

        self.downs_0 = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(4, 4)])
        self.downs_1 = nn.ModuleList([nn.MaxPool2d(2, 2)])
        self.ups_1 = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear')])
        self.ups_2 = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=4, mode='bilinear')])
        self.ups_3 = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=4, mode='bilinear')])
        self.ups_4 = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is Multi_Scale_Fully_Connected_ConvLSTM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        raw_out = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, next_hiddens = self.lstms[l](x, hiddens)
            raw_out.append(x)
            if l == 0:
                out_0_down = self.downs_0[0](x)
                out_0_down_down = self.downs_0[1](x)
                x = out_0_down
            elif l == 1:
                out_1_down = self.downs_1[0](x)
                out_1_up = self.ups_1[0](x)
                x = out_1_down + out_0_down_down
            elif l == 2:
                out_2_up = self.ups_2[0](x)
                out_2_up_up = self.ups_2[1](x)
                x = out_1_down + out_0_down_down + x
            elif l == 3:
                out_3_up = self.ups_3[0](x)
                out_3_up_up = self.ups_3[1](x)
                x = out_0_down + raw_out[1] + out_2_up + out_3_up
            elif l == 4:
                out_4_up = self.ups_4[0](x)
                x = raw_out[0] + out_1_up + out_2_up_up + out_3_up_up + out_4_up
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        return x, m, next_layer_hiddens
