from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
import os
from os import listdir
import random
import copy
from torch.utils.data import DataLoader
from skimage import measure #supports video also
import pickle
from scipy.spatial import distance
import time
import platform

from utils.Network import Network
from utils.Analyser import Analyser
from utils.io import save_network, save, load, figure_save, make_folder_results, imshow
from utils.format import hex_str2bool
from utils.WaveDataset import Create_Datasets

debug = False

logging.basicConfig(format='%(message)s',level=logging.INFO)
channels=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformVar = {"Test": transforms.Compose([
    transforms.Resize(128),    #Already 184 x 184
    transforms.CenterCrop(128),
    transforms.ToTensor(),
#     normalize
]),
    "Train": transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
#     normalize
    ])
}

### COMMON FUNCTIONS TRAIN/VAL/TEST
def initial_input(model, image_series, starting_point, num_input_frames, channels, device, training):
    """
    var           size
    image_series  [16, 100, 128, 128]
    input_frames  [16, 5, 128, 128]
    output_frame  [16, 1, 128, 128]
    target_frame  [16, 1, 128, 128]
    """
    input_frames = image_series[:, starting_point * channels:(starting_point + num_input_frames) * channels, :, :].to(device)
    output_frame = model(input_frames, mode='initial_input', training=training)
    index = starting_point + num_input_frames
    target_frame = image_series[:, index * channels:(index + 1) * channels, :, :].to(device)
    return output_frame, target_frame

def reinsert(model, image_series, starting_point, num_input_frames, num_output_frames, output_frame, target_frame, channels, device, training):
    output_frame = torch.cat((output_frame, model(output_frame[:, -num_input_frames * channels:, :, :].clone(), mode="reinsert", training=training)), dim=1)
    index = starting_point + num_output_frames + num_input_frames
    target_frame = torch.cat((target_frame, image_series[:, index * channels:(index + 1) * channels, :, :].to(device)), dim=1)
    return output_frame, target_frame

def propagate(model, image_series, starting_point, num_input_frames, n, output_frame, target_frame, channels, device, training):
    output_frame = torch.cat((output_frame, model(torch.Tensor([0]), mode="propagate", training=training)), dim=1)
    index = starting_point + n + num_input_frames
    target_frame = torch.cat((target_frame, image_series[:, index * channels:(index + 1) * channels, :, :].to(device)), dim=1)
    return output_frame, target_frame

def plot_predictions(output_frame, target_frame, channels):
    predicted = output_frame[i, -channels:, :, :].cpu().detach()
    des_target_frame = target_frame[i, -channels:, :, :].cpu().detach()
    fig = plt.figure()
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
    tar = fig.add_subplot(1, 2, 2)
    imshow(des_target_frame, title="Target %02d" % n, obj=tar)
    plt.show()


def train_epoch(model, epoch, train_dataloader, val_dataloader, num_input_frames, num_output_frames, channels, device, plot=False,):
    """
    Training of the network
    :param train: Training data
    :param val_dataloader: Validation data
    :return:
    """
    training = True
    model.train()           # initialises training stage/functions
    mean_loss = 0
    logging.info('Training: Ready to load batches')
    for batch_num, batch in enumerate(train_dataloader):
        batch_start = time.time()
        # logging.info('Batch: %d loaded in %.3f' %(batch_num, batch_time))
        mean_batch_loss = 0
        random_starting_points = random.sample(range(100 - num_input_frames - (2 * num_output_frames) - 1), 10)
        image_series = batch["image"]
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=image_series.size()[0], training=True)
            lr_scheduler.optimizer.zero_grad()
            for n in range(2 * num_output_frames):
                if n == 0:
                    output_frame, target_frame = initial_input(model, image_series, starting_point, num_input_frames, channels, device, training=training)
                elif n == num_output_frames:
                    output_frame, target_frame = reinsert(model, image_series, starting_point, num_input_frames, num_output_frames, output_frame, target_frame, channels, device, training=training)
                else:
                    output_frame, target_frame = propagate(model, image_series, starting_point, num_input_frames, n, output_frame, target_frame, channels, device, training=training)
                if plot and (i == 0) and (batch_num == 0):
                    plot_predictions(output_frame, target_frame, channels)
            loss = F.mse_loss(output_frame, target_frame)
            loss.backward()
            lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()
            if debug: break

        analyser.save_loss_batchwise(mean_batch_loss / (i + 1), batch_increment=1)
        mean_loss += loss.item()

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1,
                   len(train_dataloader), 100. * (batch_num + 1) / len(train_dataloader), loss.item(), batch_time ) )        

        if debug and batch_num > 2:
            print('break')
            break

    analyser.save_epoch_loss(mean_loss / (batch_num + 1), 1)
    val_start = time.time()
    validation_loss = validate(model, val_dataloader, num_input_frames, num_output_frames, channels, device, plot=False)
    analyser.save_validation_loss(validation_loss, 1)
    val_time = time.time() - val_start
    logging.info('Validation loss: %.6f\tTime: %.3f' % (validation_loss, val_time))



def validate(model, val_dataloader, num_input_frames, num_output_frames, channels, device, plot=False):
    """
    Validation of network (same protocol as training)
    :param val_dataloader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    training = False
    model.eval()
    overall_loss = 0
    for batch_num, batch in enumerate(val_dataloader):
        random_starting_points = random.sample(range(100 - num_input_frames - (2 * num_output_frames) - 1), 10)
        image_series = batch["image"]
        batch_loss = 0
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=image_series.size()[0], training=False)
            for n in range(2 * num_output_frames):
                if n == 0:
                    output_frame, target_frame = initial_input(model, image_series, starting_point, num_input_frames, channels, device, training=training)
                elif n == num_output_frames:
                    output_frame, target_frame = reinsert(model, image_series, starting_point, num_input_frames, num_output_frames, output_frame, target_frame, channels, device, training=training)
                else:
                    output_frame, target_frame = propagate(model, image_series, starting_point, num_input_frames, n, output_frame, target_frame, channels, device, training=training)
                if plot and (i == 0) and (batch_num == 0):
                    plot_predictions(output_frame, target_frame, channels)
            batch_loss += F.mse_loss(output_frame, target_frame).item()
        overall_loss += batch_loss / (i + 1)
        if debug: break
    val_loss = overall_loss / (batch_num + 1)
    return val_loss

# get_ipython().system('rm -rf Results/')
# get_ipython().system('rm Video_Data/.DS_Store')

nr_net = 0 

version = nr_net + 10
num_input_frames = 5
num_output_frames = 10
network_type = "7_kernel_3LSTM"

if 'Darwin' in platform.system():
    data_dir = './'
else:
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

if not os.path.isdir("./Results"):
    os.mkdir("./Results")
results_dir = "./Results/" + network_type + "_v%03d/" % version

if not os.path.isdir(results_dir):
    make_folder_results(results_dir)


# Data
filename_data = results_dir + "all_data.pickle"
if os.path.isfile(filename_data):
    logging.info('Loading datasets')
    all_data = load(filename_data)
    train_dataset = all_data["Training data"]
    val_dataset = all_data["Validation data"]
    test_dataset = all_data["Testing data"]
else:
    logging.info('Creating new datasets')
    test_dataset, val_dataset, train_dataset = Create_Datasets(
         data_dir+"Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False, channels=channels)
    all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
    save(all_data, filename_data)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)

# analyser
filename_analyser = results_dir + network_type + "_analyser.pickle" 
if os.path.isfile(filename_analyser):
    logging.info('Loading analyser')
    analyser = load(filename_analyser)
else:
    logging.info('Creating analyser')
    analyser = Analyser(results_dir)

# Model
filename_model = results_dir + "model.pt"
if os.path.isfile(filename_model):
    model = torch.load(filename_model)
else:
    model = Network(device, channels)

# Learning Rate scheduler w. optimizer
# Optimizer
optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# Add learning rate schedulers
# Decay LR by a factor of gamma every step_size epochs
scheduler_type = 'plateau'
if scheduler_type == 'step':
    gamma = 0.5
    step_size = 40
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_algorithm, step_size=step_size, gamma=gamma)
elif scheduler_type == 'plateau':
    # Reduce learning rate when a metric has stopped improving
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)

filename_metadata = results_dir + "metadata.pickle" 
meta_data_dict = {  "optimizer": optimizer_algorithm.state_dict(),
                    "scheduler_type": scheduler_type, 
                    "scheduler": lr_scheduler.state_dict()}
save(meta_data_dict, filename_metadata)

model.to(device)

if __name__ == "__main__":
    logging.info('Experiment %d' % version)
    logging.info('Start training')
    epochs=3
    for epoch in range(epochs):
        epoch_start = time.time()

        logging.info('Epoch %d' % epoch)
        train_epoch(model, epoch, train_dataloader, val_dataloader, num_input_frames, num_output_frames, channels, device, plot=False)
        """
        Here we can access analyser.validation_loss to make decisions
        """
        # Learning rate scheduler
        # perform scheduler step if independent from validation loss
        if scheduler_type == 'step':
            lr_scheduler.step()
        # perform scheduler step if Dependent on validation loss
        if scheduler_type == 'plateau':
            validation_loss = analyser.validation_loss[-1]
            lr_scheduler.step(validation_loss)
        save_network(model, filename_model)
        save(analyser, filename_analyser)

        epoch_time = time.time() - epoch_start 
        logging.info('Epoch time: %.1f' % epoch_time)

# analyser = []
# model =[]
# lr_scheduler = []
# scheduler_dict = []

# analyser.plot_loss()
# analyser.plot_loss_batchwise()
# analyser.plot_validation_loss()