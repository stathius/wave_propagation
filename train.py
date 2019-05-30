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
import scipy.ndimage as ndimage
from scipy.spatial import distance
import time
import platform

from utils.Network import Network
from utils.Analyser import Analyser
from utils.io import save_network, save, load, figure_save, make_folder_results, imshow
from utils.format import hex_str2bool
from utils.WaveDataset import Create_Datasets

logging.basicConfig(format='%(message)s',level=logging.INFO)

channels=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

import time
def train_epoch(model, epoch, train_dataloader, val_dataloader, channels, device, plot=False,):
    """
    Training of the network
    :param train: Training data
    :param val_dataloader: Validation data
    :return:
    """
    def initial_input(channels, training):
        Data = ImageSeries[:, (t0 + n) * channels:(t0 + n + input_frames) * channels, :, :].to(device)
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)
        return output, target

    def new_input(output, target, channels, training):
        output = torch.cat((output, model(
            output[:, -input_frames * channels:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def consequent_propagation(output, target, channels, training):
        output = torch.cat((output, model(torch.Tensor([0]), mode="internal", training=training)), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def plot_predictions():
        if (i == 0) & (batch_num == 0):
            predicted = output[i, -channels:, :, :].cpu().detach()
            des_target = target[i, -channels:, :, :].cpu().detach()
            fig = plt.figure()
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % n, obj=tar)
            plt.show()

    model.train()           # initialises training stage/functions
    mean_loss = 0
    logging.info('Training: Ready to load batches')
    for batch_num, batch in enumerate(train_dataloader):
        batch_start = time.time()
        # logging.info('Batch: %d loaded in %.3f' %(batch_num, batch_time))
        mean_batch_loss = 0
        sequence_starting_points = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        for i, t0 in enumerate(sequence_starting_points):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=True)
            lr_scheduler.optimizer.zero_grad()
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(channels, training=True)
                elif n == output_frames:
                    output, target = new_input(output, target, channels, training=True)
                else:
                    output, target = consequent_propagation(output, target, channels, training=True)
                if plot:
                    plot_predictions()
            loss = F.mse_loss(output, target)
            loss.backward()
            lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()


        analyser.save_loss_batchwise(mean_batch_loss / (i + 1), batch_increment=1)
        mean_loss += loss.item()

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1,
                   len(train_dataloader), 100. * (batch_num + 1) / len(train_dataloader), loss.item(), batch_time ) )        


    analyser.save_loss(mean_loss / (batch_num + 1), 1)
    validation_loss = validate(val_dataloader, channels, plot=False)
    analyser.save_validation_loss(validation_loss, 1)
    logging.info('Validation loss: %.3f ' % validation_loss)



def validate(model, val_dataloader, channels, plot=False):
    """
    Validation of network (same protocol as training)
    :param val_dataloader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    def initial_input(channels, training):
        Data = ImageSeries[:, (t0 + n) * channels:(t0 + n + input_frames) * channels, :, :].to(device)
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)
        return output, target
 
    def new_input(output, target, channels, training):
        output = torch.cat((output, model(
            output[:, -input_frames * channels:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def consequent_propagation(output, target, channels, training):
        output = torch.cat((output, model(torch.Tensor([0]), mode="internal", training=training)), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def plot_predictions():
        if (i == 0) & (batch_num == 0):
            predicted = output[i, -channels:, :, :].cpu().detach()
            des_target = target[i, -channels:, :, :].cpu().detach()
            fig = plt.figure()
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % n, obj=tar)
            plt.show()

    model.eval()
    overall_loss = 0
    for batch_num, batch in enumerate(val_dataloader):
        sequence_starting_points = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        batch_loss = 0
        for i, t0 in enumerate(sequence_starting_points):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=False)
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(channels, training=False)
                elif n == output_frames:
                    output, target = new_input(output, target, channels, training=False)
                else:
                    output, target = consequent_propagation(output, target, channels, training=False)
                if plot:
                    plot_predictions()
            batch_loss += F.mse_loss(output, target).item()
        overall_loss += batch_loss / (i + 1)
    val_loss = overall_loss / (batch_num + 1)
    return val_loss

# get_ipython().system('rm -rf Results/')
# get_ipython().system('rm Video_Data/.DS_Store')

nr_net = 0 

version = nr_net + 10
input_frames = 5
output_frames = 10
network_type = "7_kernel_3LSTM"

# Little trick to adjust path files for compatibility (I have a backup of the Main.py in case it doesn't work)
# stef_path = "/media/sg6513/DATADRIVE2/MSc/Wavebox/"
# if os.path.isfile(stef_path + "stefpc.txt"):
#     if not os.path.isdir(stef_path + "Results"):
#         os.mkdir(stef_path + "Results")
#     results_dir = stef_path + "Results/Simulation_Result_" + network_type + "_v%03d/" % version
#     maindir2 = stef_path
#     version += 200
# else:


if 'Darwin' in platform.system():
    data_dir = './'
else:
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

if not os.path.isdir("./Results"):
    os.mkdir("./Results")
results_dir = "./Results/Simulation_Result_" + network_type + "_v%03d/" % version

if not os.path.isdir(results_dir):
    make_folder_results(results_dir)


# Data
filename_data = results_dir + "all_data_" + "_v%03d.pickle" % version
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


# analyser
filename_analyser = results_dir + network_type + "_analyser_v%03d.pickle" % version
if os.path.isfile(filename_analyser):
    logging.info('Loading analyser')
    analyser = load(filename_analyser)
else:
    logging.info('Creating analyser')
    analyser = Analyser(results_dir)

# Model
filename_model = results_dir + network_type + "_model_v%03d.pt" % version
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

filename_metadata = results_dir + network_type + "_analyser_v%03d.pickle" % version
meta_data_dict = {  "optimizer": optimizer_algorithm.state_dict(),
                    "scheduler_type": scheduler_type, 
                    "scheduler": lr_scheduler.state_dict()}
save(scheduler_dict, filename_metadata)


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)

root_dir = train_dataset.root_dir
img_path = train_dataset.All_Imagesets[0]
im_list = sorted(listdir(root_dir + img_path[1]))

model.to(device)

logging.info('Experiment %d' % version)
logging.info('Start training')
epochs=1
for epoch in range(epochs):
    epoch_start = time.time()

    logging.info('Epoch %d' % epoch)
    train_epoch(model, epoch, train_dataloader, val_dataloader, channels, device, plot=False,)
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
# analyser.plot_accuracy()
# analyser.plot_loss_batchwise()
# analyser.plot_validation_loss()