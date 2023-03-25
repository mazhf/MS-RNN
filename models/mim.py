from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class MIM_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w

        self._conv_x2h_n = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_n2h_n = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_diff2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                          kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_n2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_s = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_c2h_s = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_s2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_c2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_c_m = cfg.LSTM_conv(in_channels=2 * output_channel, out_channels=output_channel,
                                       kernel_size=1, stride=1, padding=0)

        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, xt_1, m, hiddens, l):
        if hiddens is None:
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            n = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            s = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c, n, s = hiddens
        if xt_1 is None:
            xt_1 = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                               dtype=torch.float).cuda()
        if m is None:
            m = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()

        # 第一层时，n和s，D和T，虽然计算，向时间轴传递，但不使用
        x2h_n = self._conv_x2h_n(x - xt_1)
        n2h_n = self._conv_n2h_n(n)
        i_n, f_n, g_n = torch.chunk((x2h_n + n2h_n), 3, dim=1)
        o_n = self._conv_diff2o(x - xt_1)
        i_n = torch.sigmoid(i_n)
        f_n = torch.sigmoid(f_n)
        g_n = torch.tanh(g_n)
        next_n = f_n * n + i_n * g_n
        o_n = torch.sigmoid(o_n + self._conv_n2o(next_n))
        D = o_n * torch.tanh(next_n)

        x2h_s = self._conv_x2h_s(D)
        c2h_s = self._conv_c2h_s(c)
        i_s, f_s, g_s, o_s = torch.chunk((x2h_s + c2h_s), 4, dim=1)
        i_s = torch.sigmoid(i_s)
        f_s = torch.sigmoid(f_s)
        g_s = torch.tanh(g_s)
        next_s = f_s * s + i_s * g_s
        o_s = torch.sigmoid(o_s + self._conv_s2o(next_s))
        T = o_s * torch.tanh(next_s)

        x2h = self._conv_x2h(x)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        if l == 0:
            next_c = f * c + i * g
        else:
            next_c = T + i * g

        x2h_m = self._conv_x2h_m(x)
        m2h_m = self._conv_m2h_m(m)
        i_m, f_m, g_m = torch.chunk((x2h_m + m2h_m), 3, dim=1)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)
        next_m = f_m * m + i_m * g_m

        o = torch.sigmoid(o + self._conv_c2o(next_c) + self._conv_m2o(next_m))
        next_h = o * torch.tanh(self._conv_c_m(torch.cat([next_c, next_m], dim=1)))

        output = next_h
        next_hiddens = [next_h, next_c, next_n, next_s]
        return output, next_m, next_hiddens


class MIM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [MIM_Cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is MIM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
                if l != 0:
                    xt_1 = layer_hiddens[l - 1][0]
                else:
                    xt_1 = None
            else:
                hiddens = None
                xt_1 = None
            x, m, next_hiddens = self.lstm[l](x, xt_1, m, hiddens, l)
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss
