from torch import nn
import torch
from collections import OrderedDict
import random
import logging
from utils.helper_functions import convert_SBCHW_to_BSHW, convert_BSHW_to_SBCHW
from utils.plotting import plot_predictions, plot_cutthrough


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device=None, seq_len=None):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter * 4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        self.Wci = torch.zeros(1, num_filter, self._state_height, self._state_width).to(device)
        self.Wcf = torch.zeros(1, num_filter, self._state_height, self._state_width).to(device)
        self.Wco = torch.zeros(1, num_filter, self._state_height, self._state_width).to(device)
        self.Wci.requires_grad = True
        self.Wcf.requires_grad = True
        self.Wco.requires_grad = True
        self._input_channel = input_channel
        self._num_filter = num_filter
        self.device = device
        self.seq_len = seq_len
    # inputs and states should not be all none
    # inputs: S*B*C*H*W

    def forward(self, inputs=None, states=None, seq_len=None):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                            self._state_width), dtype=torch.float).to(self.device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(self.device)
        else:
            h, c = states
        if seq_len is None:
            seq_len = self.seq_len
        outputs = []
        for index in range(seq_len):
            # print(index)
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                self._state_width), dtype=torch.float).to(self.device)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        # logging.debug(input.size())
        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, seq_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
        self.seq_len = seq_len

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=self.seq_len)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i - 1], getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)))
        return input


class EncoderForecaster(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.input_dim = 'SBCHW'  # indicates the model expects inputs in the form S*B*C*H*W

    def forward(self, input):
        input = convert_BSHW_to_SBCHW(input)
        state = self.encoder(input)
        output = self.forecaster(state)
        return convert_SBCHW_to_BSHW(output)

    def get_num_input_frames(self):
        return self.encoder.rnn1.seq_len

    def get_num_output_frames(self):
        return self.forecaster.rnn3.seq_len


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


def get_convlstm_model(num_input_frames, num_output_frames, batch_size, device):    # Define encoder #
    encoder_architecture = [
        # in_channels, out_channels, kernel_size, stride, padding
        [OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]}),
         OrderedDict({'conv2_leaky_1': [64, 192, 3, 2, 1]}),
         OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]})],

        [ConvLSTMCell(input_channel=8, num_filter=64, b_h_w=(batch_size, 64, 64),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_input_frames),
         ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_input_frames),
         ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_input_frames)]
    ]

    forecaster_architecture = [
        [OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
         OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
         OrderedDict({'deconv3_leaky_1': [64, 8, 4, 2, 1],
                      'conv3_leaky_2': [8, 8, 3, 1, 1],
                      'conv3_3': [8, 1, 1, 1, 0]}), ],

        [ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_output_frames),
         ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_output_frames),
         ConvLSTMCell(input_channel=64, num_filter=64, b_h_w=(batch_size, 64, 64),
                      kernel_size=3, stride=1, padding=1, device=device, seq_len=num_output_frames)]
    ]

    encoder = Encoder(encoder_architecture[0], encoder_architecture[1]).to(device)
    forecaster = Forecaster(forecaster_architecture[0], forecaster_architecture[1], num_output_frames).to(device)
    return EncoderForecaster(encoder, forecaster)


def test_convlstm(model, dataloader, starting_point, device, score_keeper, figures_dir, show_plots, debug=False, normalize=None):
    model.eval()
    num_input_frames = model.get_num_input_frames()
    num_output_frames = model.get_num_output_frames()

    for batch_num, batch_images in enumerate(dataloader):
        batch_size = batch_images.size(0)
        image_to_plot = random.randint(0, batch_size - 1)

        # total_frames = batch_images.size()[1]
        # num_future_frames = total_frames - (starting_point + num_input_frames)
        # for future_frame_idx in range(num_future_frames):
        # target_idx = starting_point + future_frame_idx + num_input_frames

        input_frames = batch_images[:, starting_point:(starting_point + num_input_frames), :, :].clone()
        output_frames = model.forward(input_frames)
        input_end_point = starting_point + num_input_frames
        target_frames = batch_images[:, input_end_point:(input_end_point + num_output_frames), :, :]

        for batch_index in range(batch_size):
            for frame_index in range(num_output_frames):
                score_keeper.add(output_frames[batch_index, frame_index, :, :].cpu(),
                                 target_frames[batch_index, frame_index:, :, :].cpu(),
                                 frame_index, "pHash", "pHash2", "SSIM", "Own", "RMSE")

        logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(dataloader)))
        if debug:
            break
    # TODO Save more frequently
    plot_predictions(frame_index, input_frames, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots)
    plot_cutthrough(frame_index, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots, direction="Horizontal", location=None)