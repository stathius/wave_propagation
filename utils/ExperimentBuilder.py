import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm
import os
import numpy as np
import time
import logging
import utils.helper_functions as helper
from utils.storage import save_statistics


class ExperimentBuilder(nn.Module):
    def __init__(self, model, lr_scheduler, experiment_name, num_epochs, train_data, val_data, test_data, device, continue_from_epoch=-1):
        super(ExperimentBuilder, self).__init__()

        self.random_points_per_batch = 10
        self.experiment_name = experiment_name
        self.model = model
        self.device = device

        self.num_input_frames = self.model.get_num_input_frames()
        self.num_output_frames = self.model.get_num_output_frames()

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.lr_scheduler = lr_scheduler

        # Generate the directory names
        self.experiment_folder = os.path.abspath("experiments_results/"+experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_loss = np.Inf

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        #self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        # self.criterion = nn.MSELoss().to(self.device)

        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

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

        batch_images = batch_images.to(self.device)
        video_length = batch_images.size(1)
        random_starting_points = random.sample(range(video_length - self.num_input_frames - self.num_output_frames - 1), self.random_points_per_batch)

        batch_loss = 0
        for i, starting_point in enumerate(random_starting_points):
            # logging.info('Starting point: %d' %i)

            input_end_point = starting_point + self.num_input_frames
            input_frames = batch_images[:, starting_point:input_end_point, :, :].clone() #.clone().detach()
            predicted_frames = self.model.forward(input_frames) # TODO REMOVE THIS
            target_frames = batch_images[:, input_end_point:(input_end_point + self.num_output_frames), :, :]
            loss = F.mse_loss(predicted_frames, target_frames)

            if train:
                self.lr_scheduler.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
                loss.backward()  # backpropagate to compute gradients for current iter loss
                self.lr_scheduler.optimizer.step()  # do a sdg step

            batch_loss += loss.item()

        # print('batch loss ', batch_loss)
        return batch_loss / self.random_points_per_batch  # mean batch loss

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        # return state['best_val_model_idx'], state['best_val_model_acc'], state
        return state['best_val_model_idx'], state['best_val_model_loss'], state

    def run_experiment(self):
        # total_losses = {"train_loss": [], "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            # epoch_start_time = time.time()
            # current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            # current_epoch_losses = {"train_loss": [], "val_loss": []}
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for batch_num, batch_images in enumerate(self.train_data):
                    # logging.info('BATCH: %d' % batch_num )
                    batch_start_time = time.time()
                    loss = self.run_batch_iter(batch_images, train=True)
                    # current_epoch_losses["train_loss"].append(loss)
                    batch_time = time.time() - batch_start_time
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f} time: {:.1f}s".format(loss, batch_time))
            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  #
                for batch_images in self.val_data:
                    with torch.no_grad():
                        loss = self.run_batch_iter(batch_images, train=False)
                    # current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}".format(loss))
            # val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            # if val_mean_loss < self.best_val_model_loss:  # if current epoch's mean val acc is greater than the saved best val acc then
                # self.best_val_model_loss = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                # self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            # for key, value in current_epoch_losses.items():
                # total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            # total_losses['curr_epoch'].append(epoch_idx)
            # save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
            #                 stats_dict=total_losses, current_epoch=i,
            #                 continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            # out_string = "_".join(
            #     ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            # epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            # epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)

            # logging.info("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            # self.state['current_epoch_idx'] = epoch_idx
            # self.state['best_val_model_loss'] = self.best_val_model_loss
            # self.state['best_val_model_idx'] = self.best_val_model_idx
            # self.save_model(model_save_dir=self.experiment_saved_models,
            #                 # save model and best val idx and best val acc, using the model dir, model name and model idx
            #                 model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            # self.save_model(model_save_dir=self.experiment_saved_models,
            #                 # save model and best val idx and best val acc, using the model dir, model name and model idx
            #                 model_save_name="train_model", model_idx='latest', state=self.state)

        # print("Generating test set evaluation metrics")
        # self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
        #                 # load best validation model
        #                 model_save_name="train_model")
        # current_epoch_losses = {"test_loss": []}  # initialize a statistics dict
        # with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
        #     for x, y in self.test_data:  # sample batch
        #         loss = self.run_evaluation_iter(x=x, y=y)  # compute loss and accuracy by running an evaluation step
        #         current_epoch_losses["test_loss"].append(loss)  # save test loss
        #         pbar_test.update(1)  # update progress bar status
        #         pbar_test.set_description(
        #             "loss: {:.4f}".format(loss))  # update progress bar string output

        # test_losses = {key: [np.mean(value)] for key, value in
                       # current_epoch_losses.items()}  # save test set metrics in dict format
        # save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',stats_dict=test_losses, current_epoch=0, continue_from_mode=False)