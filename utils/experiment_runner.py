import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm
import os
import numpy as np
import time
import logging
from utils.io import save_json
from utils.experiment_setup import save_network
from utils.plotting import save_sequence_plots


class ExperimentRunner(nn.Module):
    def __init__(self, model, lr_scheduler, experiment_name, num_epochs, samples_per_sequence, train_data, val_data, test_data, device, dirs, continue_from_epoch, debug):
        super(ExperimentRunner, self).__init__()

        self.samples_per_sequence = samples_per_sequence
        self.experiment_name = experiment_name
        self.model = model
        self.device = device
        self.debug = debug
        self.dirs = dirs

        self.num_input_frames = self.model.get_num_input_frames()
        self.num_output_frames = self.model.get_num_output_frames()

        # if torch.cuda.device_count() > 1:
        #     self.model.to(self.device)
        #     self.model = nn.DataParallel(module=self.model)
        # else:
        self.model.to(self.device)

        self.lr_scheduler = lr_scheduler

        # Generate the directory names
        self.best_val_model_loss = np.Inf
        self.num_epochs = num_epochs

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        if continue_from_epoch != -1:
            self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_batch_iter(self, batch_images, train):
        # Expects input of Batch Size x Video Length x Height x Width
        # Returns loss per each sequence prediction
        if train:
            self.model.train()
        else:
            self.model.eval()

        video_length = batch_images.size(1)
        random_starting_points = random.sample(range(video_length - self.num_input_frames - self.num_output_frames - 1), self.samples_per_sequence)

        batch_loss = 0
        for starting_point in random_starting_points:
            input_end_point = starting_point + self.num_input_frames
            input_frames = batch_images[:, starting_point:input_end_point, :, :].clone()
            # TODO change this for a model specific function prediction or something
            output_frames = self.model.forward(input_frames)
            target_frames = batch_images[:, input_end_point:(input_end_point + self.num_output_frames), :, :]
            if self.debug:
                logging.info('EXP RUNNER out tar size %s %s' % (output_frames.size(), target_frames.size()))
            loss = F.mse_loss(output_frames, target_frames)

            if train:
                self.lr_scheduler.optimizer.zero_grad()
                loss.backward()
                self.lr_scheduler.optimizer.step()

            batch_loss += loss.item()

        return batch_loss / self.samples_per_sequence  # mean batch loss

    def run_experiment(self):
        total_losses = {"train_loss": [], "val_loss": [], "curr_epoch": []}
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            logging.info('Epoch: %d' % i)
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "val_loss": []}
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for batch_num, batch_images in enumerate(self.train_data):
                    # logging.info('BATCH: %d' % batch_num )
                    batch_images = batch_images.to(self.device)
                    batch_start_time = time.time()
                    loss = self.run_batch_iter(batch_images, train=True)
                    current_epoch_losses["train_loss"].append(loss)
                    batch_time = time.time() - batch_start_time
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f} time: {:.1f}s".format(loss, batch_time))
                    if self.debug:
                        break
            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  #
                for batch_images in self.val_data:
                    batch_images = batch_images.to(self.device)
                    with torch.no_grad():
                        loss = self.run_batch_iter(batch_images, train=False)
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}".format(loss))
                    if self.debug:
                        break

            #  get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))  #
            total_losses['curr_epoch'].append(epoch_idx)
            save_json(total_losses, os.path.join(self.dirs['logs'], 'train_val_loss.json'))
            save_network(self.model, os.path.join(self.dirs['models'], 'model_latest.pt'))

            loss_string = "Train loss: {:.4f} | Validation loss: {:.4f}".format(total_losses['train_loss'][-1], total_losses['val_loss'][-1])
            epoch_elapsed_time = "{:.4f}".format(time.time() - epoch_start_time)

            logging.info("Epoch {}:\t{}\tTime elapsed {}s".format(epoch_idx, loss_string, epoch_elapsed_time))

            # Save if best model so far
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < self.best_val_model_loss:
                logging.info('Saving a better model. Previous loss: %.4f New loss: %.4f' % (self.best_val_model_loss, val_mean_loss))
                self.best_val_model_loss = val_mean_loss
                save_network(self.model, os.path.join(self.dirs['models'], 'model_best.pt'))


def test_future_frames(model, dataloader, starting_point, num_requested_output_frames, device, score_keeper, figures_dir, debug=False, normalize=None):
    model.eval()
    input_end_point = starting_point + model.get_num_input_frames()
    with torch.no_grad():
        for batch_num, batch_images in enumerate(dataloader):
            logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(dataloader)))
            batch_images = batch_images.to(device)

            input_frames = batch_images[:, starting_point:input_end_point, :, :]
            output_frames = model.get_future_frames(input_frames, num_requested_output_frames)

            num_total_output_frames = output_frames.size(1)
            target_frames = batch_images[:, input_end_point:(input_end_point + num_total_output_frames), :, :]

            score_keeper.compare_output_target(output_frames, target_frames)
            save_sequence_plots(batch_num, output_frames, target_frames, figures_dir, normalize)

            if debug:
                print('batch_num %d\tSSIM %f' % (batch_num, score_keeper.SSIM_val[-1]))
                break