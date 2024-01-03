from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class MK_LSTM_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._conv_x2h_3 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=3, stride=1, padding=1)
        self._conv_h2h_3 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=3, stride=1, padding=1)
        self._conv_x2h_5 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=5, stride=1, padding=2)
        self._conv_h2h_5 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=5, stride=1, padding=2)

        self._conv_c2o_3 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                         kernel_size=3, stride=1, padding=1)

        self._conv_c2o_5 = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                         kernel_size=5, stride=1, padding=2)

        self._conv_out = cfg.LSTM_conv(in_channels=input_channel * 2, out_channels=output_channel,
                                       kernel_size=1, stride=1, padding=0)

        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, hiddens):
        if hiddens is None:
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            c_3 = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                              dtype=torch.float).cuda()
            c_5 = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                              dtype=torch.float).cuda()
        else:
            h, c_3, c_5 = hiddens
        x2h_3 = self._conv_x2h_3(x)
        h2h_3 = self._conv_h2h_3(h)
        i_3, f_3, g_3, o_3 = torch.chunk((x2h_3 + h2h_3), 4, dim=1)

        x2h_5 = self._conv_x2h_5(x)
        h2h_5 = self._conv_h2h_5(h)
        i_5, f_5, g_5, o_5 = torch.chunk((x2h_5 + h2h_5), 4, dim=1)

        i_3 = torch.sigmoid(i_3)
        f_3 = torch.sigmoid(f_3)
        g_3 = torch.tanh(g_3)
        next_c_3 = f_3 * c_3 + i_3 * g_3

        i_5 = torch.sigmoid(i_5)
        f_5 = torch.sigmoid(f_5)
        g_5 = torch.tanh(g_5)
        next_c_5 = f_5 * c_5 + i_5 * g_5

        o = torch.sigmoid(o_3 + o_5 + self._conv_c2o_3(next_c_3) + self._conv_c2o_5(next_c_5))
        next_h = o * torch.tanh(self._conv_out(torch.cat([next_c_3, next_c_5], dim=1)))

        ouput = next_h
        next_hiddens = [next_h, next_c_3, next_c_5]
        return ouput, next_hiddens


class MS_LSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [MK_LSTM_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                MK_LSTM_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                MK_LSTM_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                MK_LSTM_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                MK_LSTM_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                MK_LSTM_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is MS-LSTM!')

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
