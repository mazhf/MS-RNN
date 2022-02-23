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
        # self._conv_c2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
        #                                kernel_size=1, stride=1, padding=0)
        # self._conv_c2i = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
        #                                kernel_size=1, stride=1, padding=0)
        # self._conv_c2f = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
        #                                kernel_size=1, stride=1, padding=0)
        self._input_channel = input_channel
        self._output_channel = output_channel
        print('This is ConvLSTM!')

    def forward(self, x, m, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens
        if x is None:
            x = torch.zeros((h.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        x2h = self._conv_x2h(x)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        # i = torch.sigmoid(i + self._conv_c2i(c))
        # f = torch.sigmoid(f + self._conv_c2f(c))
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        # o = torch.sigmoid(o + self._conv_c2o(next_c))
        o = torch.sigmoid(o)
        next_h = o * torch.tanh(next_c)

        ouput = next_h
        next_hiddens = [next_h, next_c]
        return ouput, m, next_hiddens


class ConvLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [ConvLSTM_cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)

    def forward(self, x, m, layer_hiddens, embed, fc):
        if x is not None:
            x = embed(x)
        next_layer_hiddens = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, m, next_hiddens = self.lstm[l](x, m, hiddens)  # 运行一次生成一个convlstm
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        return x, m, next_layer_hiddens
