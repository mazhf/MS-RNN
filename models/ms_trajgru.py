import torch
from torch import nn
import sys
sys.path.append("..")
from config import cfg


def warp(input, flow):
    # input: B, C, H, W
    # flow: [B, 2, H, W]
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0  # u
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0  # v
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid, align_corners=False)
    return output


class TrajGRU_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._input_channel = input_channel
        self._output_channel = output_channel
        self.L = cfg.TrajGRU_link_num

        # x to z r h_
        self._conv_x2h = nn.Conv2d(in_channels=input_channel, out_channels=output_channel * 3, kernel_size=kernel_size, stride=stride, padding=padding)

        # inputs and hidden to flow
        self._conv_x2f = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2f = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

        # generate flow
        self._conv_flow = nn.Conv2d(in_channels=output_channel, out_channels=self.L * 2, kernel_size=kernel_size, stride=stride, padding=padding)

        # Whz, Whr, Whh for project back
        self.project = nn.Conv2d(in_channels=output_channel * self.L, out_channels=output_channel * 3, kernel_size=1, stride=1, padding=0)

        # raw paper section 3.2 Convolutional GRU
        self.LeakyReLU_flow = nn.LeakyReLU(negative_slope=0.2)
        self.LeakyReLU_h_ = nn.LeakyReLU(negative_slope=0.2)

    # Structure Generating Network y
    def _flow_generator(self, x, h):
        flows = self._conv_x2f(x) + self._conv_h2f(h)
        flows = self.LeakyReLU_flow(flows)
        flows = self._conv_flow(flows)
        flows = torch.chunk(flows, self.L, dim=1)
        return flows

    def forward(self, x, h):
        if h is None:
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width), dtype=torch.float).cuda()

        x2h = self._conv_x2h(x)  # 3 * C

        flows = self._flow_generator(x, h)
        warpped_data = []
        for l in range(len(flows)):  # L
            flow = flows[l]  # u v
            warpped_data.append(warp(h, -flow))
        warpped_data = torch.cat(warpped_data, dim=1)  # L * C
        h2h = self.project(warpped_data)  # 3 * C

        z_x, r_x, h__x = torch.chunk(x2h, 3, dim=1)
        z_h, r_h, h__h = torch.chunk(h2h, 3, dim=1)
        z = torch.sigmoid(z_x + z_h)
        r = torch.sigmoid(r_x + r_h)
        h_ = self.LeakyReLU_h_(h__x + r * h__h)

        next_h = (1 - z) * h_ + z * h
        return next_h, next_h


class MS_TrajGRU(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [TrajGRU_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                TrajGRU_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                TrajGRU_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                TrajGRU_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                TrajGRU_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                TrajGRU_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is MS-TrajGRU!')

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
