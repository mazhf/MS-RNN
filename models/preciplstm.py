from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg
from img_local_att.function import LocalAttention


class PrecipLSTM_cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_c2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_ms = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                          kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_ms2h_ms = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel * 3,
                                           kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_ms2o = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel,
                                        kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_mt = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                          kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_mt2h_mt = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel * 3,
                                           kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_mt2o = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel,
                                        kernel_size=kernel_size, stride=stride, padding=padding)

        self.c_m_ms_mt = cfg.LSTM_conv(in_channels=4 * output_channel, out_channels=output_channel, kernel_size=1,
                                       stride=1, padding=0)

        self.L_num = cfg.la_num
        self.LA = LocalAttention(inp_channels=output_channel, out_channels=output_channel, kH=self.L_num, kW=self.L_num)
        self._conv_x_x_att = cfg.LSTM_conv(in_channels=output_channel, out_channels=output_channel,
                                           kernel_size=kernel_size, stride=stride, padding=padding)

        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, x_t_1, m, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            mt = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                             dtype=torch.float).cuda()
            ms = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                             dtype=torch.float).cuda()
        else:
            h, c, ms, mt = hiddens

        if x is None:
            x = torch.zeros((h.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        if m is None:
            m = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        if x_t_1 is None:
            x_t_1 = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                                dtype=torch.float).cuda()

        # LA
        x_att = self.LA(x)
        x_att = x + self._conv_x_x_att(x_att)

        # ConvLSTM
        x2h = self._conv_x2h(x_att)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        # PredRNN
        x2h_m = self._conv_x2h_m(x_att)
        m2h_m = self._conv_m2h_m(m)
        i_m, f_m, g_m = torch.chunk((x2h_m + m2h_m), 3, dim=1)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)
        next_m = f_m * m + i_m * g_m

        # SLAM
        x2h_ms = self._conv_x2h_ms(x_att)
        ms2h_ms = self._conv_ms2h_ms(ms)
        i_ms, f_ms, g_ms = torch.chunk((x2h_ms + ms2h_ms), 3, dim=1)
        i_ms = torch.sigmoid(i_ms)
        f_ms = torch.sigmoid(f_ms)
        g_ms = torch.tanh(g_ms)
        next_ms = f_ms * ms + i_ms * g_ms

        # TDM
        x2h_mt = self._conv_x2h_mt(x - x_t_1)
        mt2h_mt = self._conv_mt2h_mt(mt)
        i_mt, f_mt, g_mt = torch.chunk((x2h_mt + mt2h_mt), 3, dim=1)
        i_mt = torch.sigmoid(i_mt)
        f_mt = torch.sigmoid(f_mt)
        g_mt = torch.tanh(g_mt)
        next_mt = f_mt * mt + i_mt * g_mt

        O = torch.sigmoid(o + self._conv_c2o(next_c) + self._conv_ms2o(next_ms)
                          + self._conv_mt2o(next_mt) + self._conv_m2o(next_m))
        next_h = O * torch.tanh(self.c_m_ms_mt(torch.cat([next_c, next_ms, next_mt, next_m], dim=1)))

        ouput = next_h
        next_hiddens = (next_h, next_c, next_ms, next_mt)

        return ouput, next_m, next_hiddens


class PrecipLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [PrecipLSTM_cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is PrecipLSTM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        x_t_1_lis = []
        for l in range(self.n_layers):
            x_t_1_lis.append(x)  # x h0 h1 ... hl-2, save inputs in t-1
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
                x_t_1 = layer_hiddens[-1][l]
            else:
                hiddens = None
                x_t_1 = None
            x, m, next_hiddens = self.lstm[l](x, x_t_1, m, hiddens)
            next_layer_hiddens.append(next_hiddens)
        next_layer_hiddens.append(x_t_1_lis)  # l+1
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss
