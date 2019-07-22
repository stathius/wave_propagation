# based on the tensorflow implementation by 'yunbo':
# https://github.com/Yunbo426/predrnn-pp/blob/master/nets/predrnn_pp.py

import torch
import torch.nn as nn
from .CausalLSTM import CausalLSTMCell
from .GHU import GHU


class PredRNNPP(nn.Module):

    def __init__(self, num_input_frames, num_output_frames, batch_size, device, use_GHU=False):
        super(PredRNNPP, self).__init__()

        self.num_output_frames = num_output_frames
        self.num_input_frames = num_input_frames
        self.device = device
        self.num_hidden = [64, 64, 64, 64]
        self.num_layers = len(self.num_hidden)

        self.lstm = nn.ModuleList()
        self.output_channels = 1
        self.conv = nn.Conv2d(in_channels=1,
                               out_channels=8,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.pool = nn.MaxPool2d(kernel_size=4)

        self.compressed_shape = [8, 31, 31]
        self.use_GHU = use_GHU
        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1]
                input_channels = 8
            else:
                num_hidden_in = self.num_hidden[i - 1]
                input_channels = self.num_hidden[i - 1]

            filter_size = 3
            num_hidden_out = self.num_hidden[i]
            new_cell = CausalLSTMCell(input_channels, filter_size, num_hidden_in, num_hidden_out, self.compressed_shape, self.device)
            self.lstm.append(new_cell)

        if self.use_GHU:
            self.ghu = GHU(filter_size=3, num_features=self.num_hidden[1], input_channels=self.num_hidden[0], device=self.device)

        self.deconv = nn.ConvTranspose2d(
            in_channels=self.num_hidden[len(self.num_hidden) - 1],
            out_channels=1,
            kernel_size=7,
            stride=4,
            padding=0,
            output_padding=1
        )

    def get_num_input_frames(self):
        return self.num_input_frames

    def get_num_output_frames(self):
        return self.num_output_frames

    def get_future_frames_belated(self, input_frames, num_total_output_frames):
        num_input_frames = self.get_num_input_frames()
        num_output_frames = self.get_num_output_frames()
        output_frames = self(input_frames, num_output_frames)

        # print('CONVLSTM OUTPUT FRAMES SIZE', output_frames.size())
        while output_frames.size(1) < num_total_output_frames:
            # print('i should not appear in training')
            if output_frames.size(1) < num_input_frames:
                keep_from_input = num_input_frames - output_frames.size(1)
                input_frames = torch.cat((input_frames[:, -keep_from_input:, :, :], output_frames), dim=1)
            else:
                input_frames = output_frames[:, -num_input_frames:, :, :].clone()
            output_frames = torch.cat((output_frames, self(input_frames, num_output_frames)), dim=1)
            # print('CONVLSTM OUTPUT FRAMES SIZE', output_frames.size())
        return output_frames[:, :num_total_output_frames, :, :]

    def get_future_frames(self, input_frames, num_total_output_frames, belated):
        if belated:
            return self.get_future_frames_belated(input_frames, num_total_output_frames)
        else:
            return self(input_frames, num_total_output_frames)

    def forward(self, input_frames, num_output_frames):
        seq_length = self.num_input_frames + num_output_frames
        cell = []
        hidden = []
        mem = None
        z_t = None
        for i in range(self.num_layers):
            cell.append(None)
            hidden.append(None)
        output = []
        x_gen = None
        # x has shape B S H W
        for t in range(seq_length):

            if t < self.num_input_frames:
                inputs = input_frames[:, t, :, :].unsqueeze(1)
            else:
                inputs = x_gen

            inputs_ = self.conv(inputs)  # to 126x126
            inputs__ = self.pool(inputs_)  # to 31x31
            # Causal LSTMs do not change dimensionality
            hidden[0], cell[0], mem = self.lstm[0].forward(inputs__, hidden[0], cell[0], mem)

            if self.use_GHU:
                z_t = self.ghu(hidden[0], z_t)
            else:
                z_t = hidden[0]
            hidden[1], cell[1], mem = self.lstm[1](z_t, hidden[1], cell[1], mem)
            for i in range(2, self.num_layers):
                hidden[i], cell[i], mem = self.lstm[i](hidden[i - 1], hidden[i], cell[i], mem)

            x_gen = self.deconv(hidden[self.num_layers - 1])  # back to 100x100
            output.append(x_gen.squeeze())
            # print('t= ', t, ' memory :', torch.cuda.max_memory_allocated())

        output = torch.stack(output[self.num_input_frames:])
        if input_frames.size(0) == 1:  # if batch size one
            output = output.unsqueeze(1)
        return output.permute(1, 0, 2, 3)
