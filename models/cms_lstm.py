from torch import nn
import torch
from torch.nn import Module, Sigmoid, Tanh, ModuleList
import sys
sys.path.append("..")
from config import cfg


class _seblock(Module):
    def __init__(self, in_dim, scale=1):
        super(_seblock, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.f_key = cfg.LSTM_conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        self.f_query = cfg.LSTM_conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        self.f_value = cfg.LSTM_conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        x = inputs

        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2
        block_size = h // self.scale

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)  # B*N*H*W*2

        v_list = torch.split(value, block_size, dim=2)
        v_locals = torch.cat(v_list, 0)
        v_list = torch.split(v_locals, block_size, dim=3)
        v_locals = torch.cat(v_list)

        q_list = torch.split(query, block_size, dim=2)
        q_locals = torch.cat(q_list, 0)
        q_list = torch.split(q_locals, block_size, dim=3)
        q_locals = torch.cat(q_list)

        k_list = torch.split(key, block_size, dim=2)
        k_locals = torch.cat(k_list, 0)
        k_list = torch.split(k_locals, block_size, dim=3)
        k_locals = torch.cat(k_list)

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.in_dim, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.in_dim, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.in_dim, -1)

            sim_map = torch.bmm(query_local, key_local)
            sim_map = self.softmax(sim_map)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            context_local = context_local.view(batch_size_new, self.in_dim, h_local, w_local, 2)
            return context_local

        context_locals = func(v_locals, q_locals, k_locals)

        b, c, h, w, _ = context_locals.shape

        context_list = torch.split(context_locals, b // self.scale, 0)
        context = torch.cat(context_list, dim=3)
        context_list = torch.split(context, b // self.scale // self.scale, 0)
        context = torch.cat(context_list, dim=2)

        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        return context + x


class seblock(_seblock):
    def __init__(self, in_dim, scale=1):
        super(seblock, self).__init__(in_dim, scale)


class SE_Block(Module):
    def __init__(self, in_dim, kernel_size, img_size):
        super(SE_Block, self).__init__()
        scales = [1, 2, 4]  # divide 16*16 feature map into 16*16*1, 8*8*4, 4*4*16, etc.
        self.group = len(scales)
        self.stages = []
        self.in_dim = in_dim
        self.padding = int((kernel_size - 1) / 2)  # in this way the output has the same size
        self.stages = ModuleList([self._make_stage(in_dim, scale) for scale in scales])
        self.conv_bn = cfg.LSTM_conv(in_dim * self.group, in_dim, kernel_size=kernel_size, padding=self.padding)
        self.w_z = cfg.LSTM_conv(in_channels=2 * in_dim, out_channels=in_dim, kernel_size=1)
        self.w_h2h = cfg.LSTM_conv(in_channels=in_dim, out_channels=in_dim * 3, kernel_size=kernel_size, stride=1, padding=self.padding)
        self.w_z2h = cfg.LSTM_conv(in_channels=in_dim, out_channels=in_dim * 3, kernel_size=kernel_size, stride=1, padding=self.padding)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def _make_stage(self, in_dim, scale):
        return seblock(in_dim, scale)

    def forward(self, h_cur, m_cur):
        feats = torch.cat([h_cur, m_cur], dim=-1)
        priors = [stage(feats) for stage in self.stages]
        context = torch.cat(priors, dim=1)
        output = self.conv_bn(context)

        z_h, z_m = torch.split(output, output.shape[-1] // 2, -1)
        z = self.w_z(torch.cat([z_h, z_m], dim=1))

        z2h = self.w_z2h(z)  # [b 3*c h w]
        h2h = self.w_h2h(h_cur)  # [b 3*c h w]

        i, g, o = torch.split(h2h + z2h, self.in_dim, dim=1)
        o = self.sigmoid(o)
        g = self.tanh(g)
        i = self.sigmoid(i)
        m_next = m_cur * (1 - i) + i * g
        h_next = m_next * o

        return h_next, m_next


class CMS_LSTM_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w

        self.convQ = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        self.convR = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                   kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._input_channel = input_channel
        self._output_channel = output_channel

        self.ce_iters = cfg.ce_iters
        self.SEBlock = SE_Block(input_channel, kernel_size, cfg.height)

    def CEBlock(self, xt, ht):
        for i in range(1, self.ce_iters + 1):
            if i % 2 == 0:
                ht = (2 * torch.sigmoid(self.convR(xt))) * ht
            else:
                xt = (2 * torch.sigmoid(self.convQ(ht))) * xt
        return xt, ht

    def forward(self, x, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens

        # CE
        x, h = self.CEBlock(x, h)

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

        # SE
        next_h, next_c = self.SEBlock(next_h, next_c)

        ouput = next_h
        next_hiddens = [next_h, next_c]
        return ouput, next_hiddens


class CMS_LSTM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        lstm = [CMS_LSTM_Cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is CMS-LSTM!')

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
