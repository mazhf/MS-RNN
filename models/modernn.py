from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class MoDeRNN_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w

        self.conv33_X = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=3, stride=1, padding=1)
        self.conv55_X = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=5, stride=1, padding=2)
        self.conv77_X = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=7, stride=1, padding=3)

        self.conv33_H = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=3, stride=1, padding=1)
        self.conv55_H = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=5, stride=1, padding=2)
        self.conv77_H = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                      kernel_size=7, stride=1, padding=3)

        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._input_channel = input_channel
        self._output_channel = output_channel

        self.iters = cfg.ce_iters

    def DCB(self, xt, ht):
        for i in range(1, self.iters + 1):
            if i % 2 == 0:
                x33 = self.conv33_X(xt)
                x55 = self.conv55_X(xt)
                x77 = self.conv77_X(xt)
                x = (x33 + x55 + x77) / 3.0
                ht = 2 * torch.sigmoid(x) * ht
            else:
                h33 = self.conv33_H(ht)
                h55 = self.conv55_H(ht)
                h77 = self.conv77_H(ht)
                h = (h33 + h55 + h77) / 3.0
                xt = 2 * torch.sigmoid(h) * xt
        return xt, ht

    def forward(self, x, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens

        # DCB
        x, h = self.DCB(x, h)

        # ConvLSTM
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


class MoDeRNN(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [MoDeRNN_Cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is MoDeRNN!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, next_hiddens = self.lstm[l](x, hiddens)
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss
