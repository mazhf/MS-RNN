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
        print('This is Multi Scale ConvLSTM!')

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
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        o = torch.sigmoid(o)
        next_h = o * torch.tanh(next_c)

        ouput = next_h
        next_hiddens = [next_h, next_c]
        return ouput, m, next_hiddens


class Multi_Scale_ConvLSTM(nn.Module):
    """Unet: like Unet use maxpooling and bilinear for downsampling and upsampling"""
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

        self.downs = [nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)]
        self.ups = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')]

        concat = [nn.Conv2d(input_channel * 2, output_channel, 1, 1, 0),
                  nn.Conv2d(input_channel * 2, output_channel, 1, 1, 0)]
        self.concat = nn.ModuleList(concat)

    def forward(self, x, m, layer_hiddens, embed, fc):
        if x is not None:
            x = embed(x)
        next_layer_hiddens = []
        out = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, m, next_hiddens = self.lstm[l](x, m, hiddens)  # 运行一次生成一个convlstm
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
            elif l == 1:
                x = self.downs[1](x)
            elif l == 3:
                x = torch.cat([self.ups[0](x), out[1]], dim=1)
                x = self.concat[0](x)
            elif l == 4:
                x = torch.cat([self.ups[1](x), out[0]], dim=1)
                x = self.concat[1](x)
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        return x, m, next_layer_hiddens


# class Unet_ConvLSTM(nn.Module):
#     """Unet: Unlike Unet use maxpooling and bilinear for downsampling and upsampling, here we use conv and deconv"""
#     def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
#         super().__init__()
#         self.n_layers = cfg.LSTM_layers
#         B, H, W = b_h_w
#         lstms = [ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
#                  ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
#                  ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
#                  ConvLSTM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
#                  ConvLSTM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
#                  ConvLSTM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
#         downs = [nn.Conv2d(input_channel, output_channel, 3, 2, 1),
#                  nn.Conv2d(input_channel, output_channel, 3, 2, 1)]
#         ups = [nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1),
#                nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1)]
#         concat = [nn.Conv2d(input_channel * 2, output_channel, 1, 1, 0),
#                   nn.Conv2d(input_channel * 2, output_channel, 1, 1, 0)]
#
#         self.lstms = nn.ModuleList(lstms)
#         self.downs = nn.ModuleList(downs)
#         self.ups = nn.ModuleList(ups)
#         self.concat = nn.ModuleList(concat)
#
#     def forward(self, x, m, layer_hiddens, embed, fc):
#         if x is not None:
#             x = embed(x)
#         next_layer_hiddens = []
#         out = []
#         for l in range(self.n_layers * 2):
#             if layer_hiddens is not None:
#                 hiddens = layer_hiddens[l]
#             else:
#                 hiddens = None
#             x, m, next_hiddens = self.lstms[l](x, m, hiddens)  # 运行一次生成一个convlstm
#             out.append(x)
#             if l == 0:
#                 x = self.downs[0](x)
#             elif l == 1:
#                 x = self.downs[1](x)
#             elif l == 3:
#                 x = torch.cat([self.ups[0](x), out[1]], dim=1)
#                 x = self.concat[0](x)
#             elif l == 4:
#                 x = torch.cat([self.ups[1](x), out[0]], dim=1)
#                 x = self.concat[1](x)
#             next_layer_hiddens.append(next_hiddens)
#
#         x = fc(x)
#         return x, m, next_layer_hiddens
