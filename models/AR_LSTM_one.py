import torch
import torch.nn as nn


class AR_LSTM(nn.Module):
    """
    The network structure
    """
    def __init__(self, num_input_frames, num_output_frames, reinsert_frequency, device):
        super(AR_LSTM, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.device = device
        self.reinsert_frequency = reinsert_frequency
        self.NUM_OUTPUT_FRAMES_PER_ITER = 1  # the model outputs one frame at a time
        self.LSTM_SIZE = 1000
        # self.reset_hidden(2)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.num_input_frames, 60, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Conv2d(60, 120, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.Conv2d(120, 240, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.Conv2d(240, 480, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=480),
            nn.Tanh(),
            nn.Dropout2d(0.25)
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(30720, 1000),
            nn.Tanh(),
            nn.Dropout(0.25)
        )

        self.decoder_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1000, 30720)
        )

        self.decoder_conv = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose2d(480, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.ConvTranspose2d(240, 120, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.ConvTranspose2d(120, 60, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(60, self.NUM_OUTPUT_FRAMES_PER_ITER, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_initial_input = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)
        # self.LSTM_propagation = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)
        # self.LSTM_reinsert = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)

    def get_num_input_frames(self):
        return self.num_input_frames

    def get_num_output_frames(self):
        return self.num_output_frames

    def forward(self, x, mode="initial_input"):
        if (mode == "initial_input") or (mode == 'reinsert'):
            x = self.encoder_conv(x)
            self.org_size = x.size()
            x = x.view(-1, 30720)
            x = self.encoder_linear(x)
            if mode == "initial_input":
                self.h0, self.c0 = self.LSTM_initial_input(x, (self.h0, self.c0))
            elif mode == "reinsert":
                self.h0, self.c0 = self.LSTM_initial_input(x, (self.h0, self.c0))
        elif mode == "propagate":
            self.h0, self.c0 = self.LSTM_initial_input(self.h0, (self.h0, self.c0))
        x = self.h0.clone()
        x = self.decoder_linear(x)
        x = x.view(self.org_size)
        x = self.decoder_conv(x)
        return x

    def reset_hidden(self, batch_size):
        self.h0 = torch.zeros((batch_size, self.LSTM_SIZE)).to(self.device)
        self.c0 = torch.zeros((batch_size, self.LSTM_SIZE)).to(self.device)

    def forward_many(self, input_frames, num_total_output_frames):
        self.reset_hidden(batch_size=input_frames.size(0))
        for future_frame_idx in range(num_total_output_frames):
            if future_frame_idx == 0:
                output_frames = self(input_frames, mode='initial_input')
            elif (future_frame_idx % self.reinsert_frequency) == 0:
                input_frames = output_frames[:, -self.num_input_frames:, :, :].clone()
                output_frames = torch.cat((output_frames, self(input_frames, mode="reinsert")), dim=1)
            else:
                output_frames = torch.cat((output_frames, self(torch.Tensor([0]), mode="propagate")), dim=1)
        return output_frames

    def get_future_frames_belated(self, input_frames, num_total_output_frames):
        num_input_frames = self.get_num_input_frames()
        num_output_frames = self.get_num_output_frames()
        output_frames = self.forward_many(input_frames, num_output_frames)

        while output_frames.size(1) < num_total_output_frames:
            if output_frames.size(1) < num_input_frames:
                keep_from_input = num_input_frames - output_frames.size(1)
                input_frames = torch.cat((input_frames[:, -keep_from_input:, :, :], output_frames), dim=1)
            else:
                input_frames = output_frames[:, -num_input_frames:, :, :].clone()
            output_frames = torch.cat((output_frames, self.forward_many(input_frames, num_output_frames)), dim=1)
        return output_frames[:, :num_total_output_frames, :, :]

    def get_future_frames(self, input_frames, num_total_output_frames, belated):
        if belated:
            return self.get_future_frames_belated(input_frames, num_total_output_frames)
        else:
            return self.forward_many(input_frames, num_total_output_frames)
a