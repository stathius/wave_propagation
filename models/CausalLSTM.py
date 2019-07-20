# Based on the tensorflow implementation by yunbo:
# https://github.com/Yunbo426/predrnn-pp/blob/master/layers/CausalLSTMCell.py
import torch
import torch.nn as nn


class CausalLSTMCell(nn.Module):
    def __init__(self, input_channels, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, device, forget_bias=1.0, initializer=0.001):
        super().__init__()

        self.device = device
        # self.layer_name = layer_name
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.height = seq_shape[1]
        self.width = seq_shape[2]
        # self.layer_norm = tln
        self._forget_bias = forget_bias

        ###hidden state has similar spatial struture as inputs, we simply concatenate them on the feature dimension
        self.conv_h = nn.Conv2d(in_channels=self.num_hidden,
                                out_channels=self.num_hidden * 4,  ##lstm has four gates
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.conv_c = nn.Conv2d(in_channels=self.num_hidden,
                                out_channels=self.num_hidden * 3,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.conv_m = nn.Conv2d(in_channels=self.num_hidden_in,
                                out_channels=self.num_hidden * 3,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.conv_x = nn.Conv2d(in_channels=self.input_channels,
                                out_channels=self.num_hidden * 7,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.conv_o = nn.Conv2d(in_channels=self.num_hidden, out_channels=self.num_hidden,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.conv_1_1 = nn.Conv2d(in_channels=self.num_hidden * 2, out_channels=self.num_hidden,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, h, c, m):
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros([batch_size, self.num_hidden, self.height, self.width]).to(self.device)
        if c is None:
            c = torch.zeros([batch_size, self.num_hidden, self.height, self.width]).to(self.device)
        if m is None:
            m = torch.zeros([batch_size, self.num_hidden_in, self.height, self.width]).to(self.device)

        h_cc = self.conv_h(h)
        c_cc = self.conv_c(c)
        m_cc = self.conv_m(m)

        i_h, g_h, f_h, o_h = torch.chunk(h_cc, 4, dim=1)
        i_c, g_c, f_c = torch.chunk(c_cc, 3, dim=1)
        i_m, f_m, m_m = torch.chunk(m_cc, 3, dim=1)

        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)

        else:
            x_cc = self.conv_x(x)
            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.chunk(x_cc, 7, dim=1)
            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g
        # c2m = self.conv_h(c_new)
        i_c, g_c, f_c, o_c = torch.chunk(self.conv_h(c_new), 4, dim=1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)

        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.conv_o(m_new)

        if x is None:
            o = torch.tanh(o_h + o_c + o_m)
        else:
            o = torch.tanh(o_x + o_h + o_c + o_m)

        cell = torch.cat((c_new, m_new), 1)
        cell = self.conv_1_1(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new