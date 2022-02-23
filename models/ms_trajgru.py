import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
from config import cfg


class activation:  # 使用的是不反向传播的激活函数，torch.nn.functional.F
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)


def warp(input, flow):
    # input: B, C, H, W
    # flow: [B, 2, H, W]
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    if cfg.data_type == np.float32:
        grid = torch.cat((xx, yy), 1).float()
    elif cfg.data_type == np.float16:
        grid = torch.cat((xx, yy), 1).half()
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0  # u
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0  # v
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid, align_corners=False)
    return output


class BaseConvRNN(nn.Module):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h) \
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                            // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0


class TrajGRU_cell(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride, padding, zoneout=0.0,
                 L=cfg.TrajGRU_link_num, i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1), act_type=RNN_ACT_TYPE):
        super(TrajGRU_cell, self).__init__(num_filter=num_filter,
                                           b_h_w=b_h_w,
                                           h2h_kernel=h2h_kernel,
                                           h2h_dilate=h2h_dilate,
                                           i2h_kernel=i2h_kernel,
                                           i2h_pad=i2h_pad,
                                           i2h_stride=i2h_stride,
                                           act_type=act_type,
                                           prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout
        self.B, self.H, self.W = b_h_w
        # 对应 wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_channel,
                             out_channels=self._num_filter * 3,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=32,
                                    out_channels=self._L * 2,
                                    kernel_size=(5, 5),
                                    stride=1,
                                    padding=(2, 2))

        # 对应 hh, hz, hr，为 1 * 1 的卷积核
        self.ret = nn.Conv2d(in_channels=self._num_filter * self._L,
                             out_channels=self._num_filter * 3,
                             kernel_size=(1, 1),
                             stride=1)
        print('This is Multi Scale TrajGRU!')

    # inputs: B*C*H*W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    # inputs 和 states 不同时为空
    # inputs: S*B*C*H*W
    def forward(self, x, m, prev_h):
        if prev_h is None:
            prev_h = torch.zeros((x.shape[0], self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).cuda()

        if x is not None:
            i2h = self.i2h(x)
            i2h_slice = torch.split(i2h, self._num_filter, dim=1)
        else:
            i2h_slice = None

        if x is not None:
            flows = self._flow_generator(x, prev_h)
        else:
            flows = self._flow_generator(None, prev_h)

        warpped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            warpped_data.append(warp(prev_h, -flow))
        warpped_data = torch.cat(warpped_data, dim=1)
        h2h = self.ret(warpped_data)
        h2h_slice = torch.split(h2h, self._num_filter, dim=1)
        if i2h_slice is not None:
            reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
            update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
            new_mem = self._act_type(i2h_slice[2] + reset_gate * h2h_slice[2])
        else:
            reset_gate = torch.sigmoid(h2h_slice[0])
            update_gate = torch.sigmoid(h2h_slice[1])
            new_mem = self._act_type(reset_gate * h2h_slice[2])
        next_h = update_gate * prev_h + (1 - update_gate) * new_mem
        if self._zoneout > 0.0:
            mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
            next_h = torch.where(mask, next_h, prev_h)

        return next_h, m, next_h


class Multi_Scale_TrajGRU(nn.Module):
    """Unet: like Unet use maxpooling and bilinear for downsampling and upsampling"""

    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [TrajGRU_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                TrajGRU_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                TrajGRU_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                TrajGRU_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                TrajGRU_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                TrajGRU_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
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
