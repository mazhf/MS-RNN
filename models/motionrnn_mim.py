from torch import nn
import torch
from models.trajgru import warp
import sys
sys.path.append("..")
from config import cfg


def Warp(input, flow):
    # input: b c/4 h/2 w/2
    B, C, H, W = flow.shape  # b 2*k**2 h/2 w/2
    flow = flow.reshape(C // 2, B, 2, H, W)  # k**2 b 2 h/2 w/2
    output = []
    for i in range(C // 2):
        warpped_data = warp(input, flow[i, ...])  # b c/4 h/2 w/2
        output.append(warpped_data)  # k**2 b c/4 h/2 w/2
    output = torch.stack(output)  # k**2 b c/4 h/2 w/2
    output = output.permute(1, 2, 3, 4, 0)  # b c/4 h/2 w/2 k**2
    return output


class MotionRNN_cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()

        self.alpha = 0.5
        self.k = 3
        self._input_channel = input_channel
        self._output_channel = output_channel

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

        self._conv_enc = cfg.LSTM_conv(in_channels=input_channel, out_channels=input_channel // 4,
                                       kernel_size=2, stride=2, padding=0)

        self._conv_u = cfg.LSTM_conv(in_channels=input_channel // 4 + 2 * self.k ** 2,
                                     out_channels=2 * self.k ** 2,
                                     kernel_size=1, stride=1, padding=0)
        self._conv_r = cfg.LSTM_conv(in_channels=input_channel // 4 + 2 * self.k ** 2,
                                     out_channels=2 * self.k ** 2,
                                     kernel_size=1, stride=1, padding=0)
        self._conv_z = cfg.LSTM_conv(in_channels=input_channel // 4 + 2 * self.k ** 2,
                                     out_channels=2 * self.k ** 2,
                                     kernel_size=1, stride=1, padding=0)

        self._conv_hm = cfg.LSTM_conv(in_channels=input_channel // 4, out_channels=self.k ** 2,
                                      kernel_size=1, stride=1, padding=0)

        self._conv_dec = cfg.LSTM_conv(in_channels=input_channel // 4 * self.k ** 2,
                                       out_channels=input_channel // 4, kernel_size=1, stride=1, padding=0)
        self._deconv_dec = cfg.LSTM_deconv(in_channels=input_channel // 4, out_channels=input_channel,
                                           kernel_size=4, stride=2, padding=1)

        self._conv_g = cfg.LSTM_conv(in_channels=input_channel * 2, out_channels=output_channel,
                                     kernel_size=1, stride=1, padding=0)

    def MotionGRU(self, H, F, D):
        # Encoder
        # H: b c h w
        H_enc = self._conv_enc(H)  # b c/4 h/2 w/2

        # Transient
        u = torch.sigmoid(self._conv_u(torch.cat([H_enc, F], dim=1)))  # b 2*k**2 h/2 w/2
        r = torch.sigmoid(self._conv_r(torch.cat([H_enc, F], dim=1)))  # b 2*k**2 h/2 w/2
        z = torch.tanh(self._conv_z(torch.cat([H_enc, r * F], dim=1)))  # b 2*k**2 h/2 w/2
        F_trans = u * z + (1 - u) * F  # b 2*k**2 h/2 w/2

        # Trend
        next_D = D + self.alpha * (F - D)  # b 2*k**2 h/2 w/2

        # update F
        next_F = F_trans + next_D  # b 2*k**2 h/2 w/2

        # Broadcast and Warp
        m = torch.sigmoid(self._conv_hm(H_enc))  # b k**2 h/2 w/2
        m = m.permute(0, 2, 3, 1)  # b h/2 w/2 k**2
        m = m.unsqueeze(1)  # b 1 h/2 w/2 k**2
        H_warp = m * Warp(H_enc, next_F)  # b c/4 h/2 w/2 k**2

        # Decoder
        H_warp = H_warp.reshape(H_warp.shape[0], -1, H_warp.shape[2], H_warp.shape[3])  # b c/4 * k**2 h/2 w/2
        H_dec = self._conv_dec(H_warp)  # b c/4 h/2 w/2
        H_dec = self._deconv_dec(H_dec)  # b c h w

        # gate
        g = torch.sigmoid(self._conv_g(torch.cat([H_dec, H], dim=1)))  # b c h w
        X = g * H + (1 - g) * H_dec  # b c h w
        return X, next_F, next_D

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
            F = torch.zeros((x.shape[0], 2 * self.k ** 2, self._state_height // 2, self._state_width // 2),
                            dtype=torch.float).cuda()
            D = torch.zeros((x.shape[0], 2 * self.k ** 2, self._state_height // 2, self._state_width // 2),
                            dtype=torch.float).cuda()
        else:
            h, c, n, s, F, D = hiddens
        if xt_1 is None:
            xt_1 = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                               dtype=torch.float).cuda()
        if m is None:
            m = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()

        # 交换了block和motiongru的位置，确保MIM差分的输入是同一层的
        # MIM 第一层时，n和s，D和T，虽然计算，向时间轴传递，但不使用
        x2h_n = self._conv_x2h_n(x - xt_1)
        n2h_n = self._conv_n2h_n(n)
        i_n, f_n, g_n = torch.chunk((x2h_n + n2h_n), 3, dim=1)
        o_n = self._conv_diff2o(x - xt_1)
        i_n = torch.sigmoid(i_n)
        f_n = torch.sigmoid(f_n)
        g_n = torch.tanh(g_n)
        next_n = f_n * n + i_n * g_n
        o_n = torch.sigmoid(o_n + self._conv_n2o(next_n))
        Dif = o_n * torch.tanh(next_n)

        x2h_s = self._conv_x2h_s(Dif)
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

        # MotionGRU
        X, next_F, next_D = self.MotionGRU(next_h, F, D)

        # Motion Highway
        if l != cfg.LSTM_layers - 1:
            next_h = X + (1 - o) * x

        ouput = next_h
        next_hiddens = [next_h, next_c, next_n, next_s, next_F, next_D]
        return ouput, next_m, next_hiddens


class MotionRNN_MIM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [MotionRNN_cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is MotionRNN-MIM!')

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
        return x, m, next_layer_hiddens
