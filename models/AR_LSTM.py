import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import logging
import time
from utils.plotting import plot_predictions, plot_cutthrough


class AR_LSTM(nn.Module):
    """
    The network structure
    """
    def __init__(self, num_input_frames, num_output_frames, device):
        super(AR_LSTM, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_output_frames = 1  # num_output_frames # It should be set to 1
        self.device = device
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
            nn.ConvTranspose2d(60, self.num_output_frames, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_initial_input = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)
        self.LSTM_propagation = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)
        self.LSTM_reinsert = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)

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

    def reset_hidden(self, batch_size, training=False):
        # TODO user random values?
        self.h0 = torch.zeros((batch_size, 1000), requires_grad=training).to(self.device)
        self.c0 = torch.zeros((batch_size, 1000), requires_grad=training).to(self.device)


def initial_input(model, input_frames, batch_images, target_idx, device, training):
    """
    # Uses the first N frames of the batch starting from the random starting_point
    var           size
    batch_images  [16, 100, 128, 128]
    input_frames  [16, 5, 128, 128]
    output_frames  [16, 1, 128, 128]
    target_frames  [16, 1, 128, 128]
    """
    output_frames = model(input_frames.to(device), mode='initial_input', training=training)
    target_frames = batch_images[:, target_idx:(target_idx + 1), :, :].to(device)
    return output_frames, target_frames


def reinsert(model, input_frames, output_frames, target_frames, batch_images, target_idx, device, training):
    # It will insert the last N predictions as an input
    output_frames = torch.cat((output_frames, model(input_frames, mode="reinsert", training=training)), dim=1).to(device)
    target_frames = torch.cat((target_frames, batch_images[:, target_idx:(target_idx + 1), :, :].to(device)), dim=1)
    return output_frames, target_frames


def propagate(model, output_frames, target_frames, batch_images, target_idx, device, training):
    # This doesn't take any input, just propagates the LSTM internal state once
    output_frames = torch.cat((output_frames, model(torch.Tensor([0]), mode="propagate", training=training)), dim=1).to(device)
    target_frames = torch.cat((target_frames, batch_images[:, target_idx:(target_idx + 1), :, :].to(device)), dim=1)
    return output_frames, target_frames


def run_iteration(model, lr_scheduler, epoch, dataloader, num_input_frames, num_output_frames, reinsert_frequency, device, analyser, training, show_plots=False, debug=False):
    samples_per_sequence = 10
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
        for sp_idx, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=batch_images.size()[0], training=training)
            for future_frame_idx in range(num_output_frames):
                target_idx = starting_point + future_frame_idx + num_input_frames
                if future_frame_idx == 0:
                    input_frames = batch_images[:, starting_point:(starting_point + num_input_frames), :, :].clone()
                    output_frames, target_frames = initial_input(model, input_frames, batch_images, target_idx, device, training)
                elif future_frame_idx == reinsert_frequency:
                    input_frames = output_frames[:, -num_input_frames:, :, :].clone()
                    output_frames, target_frames = reinsert(model, input_frames, output_frames, target_frames, batch_images, target_idx, device, training)
                else:
                    output_frames, target_frames = propagate(model, output_frames, target_frames, batch_images, target_idx, device, training)
            loss = F.mse_loss(output_frames, target_frames)
            batch_loss += loss.item()

            if training:
                lr_scheduler.optimizer.zero_grad()
                loss.backward()
                lr_scheduler.optimizer.step()

            if debug:
                break
        mean_batch_loss = batch_loss / (sp_idx + 1)
        analyser.save_loss_batchwise(mean_batch_loss, batch_increment=1)
        total_loss += mean_batch_loss

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1, len(dataloader), 100. * (batch_num + 1) / len(dataloader), mean_batch_loss, batch_time))

        if debug:
            break
    mean_loss = total_loss / (batch_num + 1)
    # plot_predictions(batch_num, output_frames, target_frames, show_plots)
    return mean_loss


def test_ar_lstm(model, dataloader, starting_point, num_input_frames, reinsert_frequency, device, score_keeper, figures_dir, show_plots, debug=False, normalize=None):
    model.eval()
    NUM_CHANNELS = 1
    training = False

    for batch_num, batch_images in enumerate(dataloader):
        batch_size = batch_images.size()[0]
        model.reset_hidden(batch_size=batch_images.size()[0], training=training)
        image_to_plot = random.randint(0, batch_size - 1)

        total_frames = batch_images.size()[1]
        num_future_frames = total_frames - (starting_point + num_input_frames)
        for future_frame_idx in range(num_future_frames):
            target_idx = starting_point + future_frame_idx + num_input_frames
            if future_frame_idx == 0:
                input_frames = batch_images[:, starting_point:(starting_point + num_input_frames), :, :].clone()
                output_frames, target_frames = initial_input(model, input_frames, batch_images, target_idx, device, training)
            elif future_frame_idx % reinsert_frequency == 0:
                input_frames = output_frames[:, -num_input_frames:, :, :].clone()
                output_frames, target_frames = reinsert(model, input_frames, output_frames, target_frames, batch_images, target_idx, device, training)
            else:
                output_frames, target_frames = propagate(model, output_frames, target_frames, batch_images, target_idx, device, training)
                # output & target_frames size is [batches, * (n + 1), 128, 128]
            if debug:
                print('batch_num %d\tfuture_frame_idx %d' % (batch_num, future_frame_idx))

            for ba in range(output_frames.size()[0]):
                score_keeper.add(output_frames[ba, -NUM_CHANNELS:, :, :].cpu(),
                                 target_frames[ba, -NUM_CHANNELS:, :, :].cpu(),
                                 future_frame_idx, "pHash", "pHash2", "SSIM", "Own", "RMSE")

            # if  batch_num == 1 and (((future_frame_idx + 1) % plot_frequency) == 0 or (future_frame_idx == 0)):

        logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(dataloader)))
        if debug:
            break
    # TODO Save more frequently
    plot_predictions(future_frame_idx, input_frames, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots)
    plot_cutthrough(future_frame_idx, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots, direction="Horizontal", location=None)