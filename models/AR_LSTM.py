import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import logging
import time


class AR_LSTM(nn.Module):
    """
    The network structure
    """
    def __init__(self, num_input_frames, reinsert_frequency, device):
        super(AR_LSTM, self).__init__()
        self.num_input_frames = num_input_frames
        self.device = device
        self.reinsert_frequency = reinsert_frequency
        self.NUM_OUTPUT_FRAMES = 1  # the model outputs one frame at a time
        self.LSTM_SIZE = 1000
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
            nn.ConvTranspose2d(60, self.NUM_OUTPUT_FRAMES, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_initial_input = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)
        self.LSTM_propagation = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)
        self.LSTM_reinsert = nn.LSTMCell(input_size=self.LSTM_SIZE, hidden_size=self.LSTM_SIZE, bias=True)

    def get_num_input_frames(self):
        return self.num_input_frames

    def get_num_output_frames(self):
        return self.NUM_OUTPUT_FRAMES

    def forward(self, x, mode="initial_input"):
        if "initial_input" in mode:
            x = self.encoder_conv(x)
            self.org_size = x.size()
            x = x.view(-1, 30720)
            x = self.encoder_linear(x)
            if mode == "initial_input":
                self.h0, self.c0 = self.LSTM_initial_input(x, (self.h0, self.c0))
            elif mode == "reinsert":
                self.h0, self.c0 = self.LSTM_reinserting(x, (self.h0, self.c0))
        elif mode == "propagate":
            self.h0, self.c0 = self.LSTM_propagation(self.h0, (self.h0, self.c0))
        x = self.h0.clone()
        x = self.decoder_linear(x)
        x = x.view(self.org_size)
        x = self.decoder_conv(x)
        return x

    def reset_hidden(self, batch_size):
        self.h0 = torch.zeros((batch_size, self.LSTM_SIZE)).to(self.device)
        self.c0 = torch.zeros((batch_size, self.LSTM_SIZE)).to(self.device)

    def get_future_frames(self, input_frames, num_requested_output_frames):
        self.reset_hidden(batch_size=input_frames.size(0))
        for future_frame_idx in range(num_requested_output_frames):
            if future_frame_idx == 0:
                output_frames = self(input_frames, mode='initial_input')
            elif future_frame_idx == self.reinsert_frequency:
                input_frames = output_frames[:, -self.num_input_frames:, :, :].clone()
                output_frames = torch.cat((output_frames, self(input_frames, mode="reinsert")), dim=1)
            else:
                output_frames = torch.cat((output_frames, self(torch.Tensor([0]), mode="propagate")), dim=1)
        return output_frames


def run_iteration(model, lr_scheduler, epoch, dataloader, num_input_frames, num_output_frames, reinsert_frequency, samples_per_sequence, device, analyser, training=False, debug=False):
    if training:
        model.train()
    else:
        model.eval()           # initialises training stage/functions
    total_loss = 0
    for batch_num, batch_images in enumerate(dataloader):
        batch_start = time.time()
        batch_loss = 0
        sequence_length = batch_images.size(1)
        random_starting_points = random.sample(range(sequence_length - num_input_frames - num_output_frames - 1), samples_per_sequence)
        # print('RANDOM POINTS: ',random_starting_points)

        for sp_idx, starting_point in enumerate(random_starting_points):
            target_index = starting_point + num_input_frames
            input_frames = batch_images[:, starting_point:target_index, :, :].clone()
            output_frames = model.get_future_frames(input_frames, num_output_frames)
            target_frames = batch_images[:, target_index:(target_index + num_output_frames), :, :]

            # print('ar size out tar ', output_frames.size(), target_frames.size())
            loss = F.mse_loss(output_frames, target_frames)
            # print('idx, loss: ', sp_idx, loss.item())
            batch_loss += loss.item()

            if training:
                lr_scheduler.optimizer.zero_grad()
                loss.backward()
                lr_scheduler.optimizer.step()

        mean_batch_loss = batch_loss / (sp_idx + 1)
        analyser.save_loss_batchwise(mean_batch_loss, batch_increment=1)
        total_loss += mean_batch_loss

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1, len(dataloader), 100. * (batch_num + 1) / len(dataloader), mean_batch_loss, batch_time))

        if debug:
            break
    mean_loss = total_loss / (batch_num + 1)
    return mean_loss