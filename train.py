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
def train(model, epoch, train_dataloader, val_dataloader, channels, device, plot=False,):
    """
    Training of the network
    :param epoch: Which epoch are you on
    :param train: Training data
    :param val_dataloader: Validation data
    :return:
    """
    ### add grayscayle or rgb flag to gain speed
    def initial_input(training, channels):
        Data = ImageSeries[:, (t0 + n) * channels:(t0 + n + input_frames) * channels, :, :].to(device)
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)
        return output, target

    def new_input(output, target, training, channels):
        output = torch.cat((output, model(
            output[:, -input_frames * channels:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def consequent_propagation(output, target, training, channels):
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
        Starting_times = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        for i, t0 in enumerate(Starting_times):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=True)
            exp_lr_scheduler.optimizer.zero_grad()
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(training=True, channels=channels)
                elif n == output_frames:
                    output, target = new_input(output, target, training=True, channels=channels)
                else:
                    output, target = consequent_propagation(output, target, training=True, channels=channels)
                if plot:
                    plot_predictions()
            loss = F.mse_loss(output, target)
            loss.backward()
            exp_lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()


        analyser.save_loss_batchwise(mean_batch_loss / (i + 1), 1)
        mean_loss += loss.item()

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1,
                   len(train_dataloader), 100. * (batch_num + 1) / len(train_dataloader), loss.item(), batch_time ) ) 
        break       


    analyser.save_loss(mean_loss / (batch_num + 1), 1)
    validation_loss = validate(val_dataloader, channels, plot=False)
    analyser.save_validation_loss(validation_loss, 1)
    logging.info('Validation loss: %.3f ' % validation_loss)



def validate(val_dataloader, channels, plot=False):
    """
    Validation of network (same protocol as training)
    :param val_dataloader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    def initial_input(training):
        Data = ImageSeries[:, (t0 + n) * channels:(t0 + n + input_frames) * channels, :, :].to(device)
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)
        return output, target
 
    def new_input(output, target, training):
        output = torch.cat((output, model(
            output[:, -input_frames * channels:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * channels:(t0 + n + input_frames + 1) * channels, :, :].to(device)), dim=1
        )
        return output, target

    def consequent_propagation(output, target, training):
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
        Starting_times = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        batch_loss = 0
        for i, t0 in enumerate(Starting_times):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=False)
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(training=False)
                elif n == output_frames:
                    output, target = new_input(output, target, training=False)
                else:
                    output, target = consequent_propagation(output, target, training=False)
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
Type_Network = "7_kernel_3LSTM"
DataGroup = "LSTM"


# Little trick to adjust path files for compatibility (I have a backup of the Main.py in case it doesn't work)
# stef_path = "/media/sg6513/DATADRIVE2/MSc/Wavebox/"
# if os.path.isfile(stef_path + "stefpc.txt"):
#     if not os.path.isdir(stef_path + "Results"):
#         os.mkdir(stef_path + "Results")
#     maindir1 = stef_path + "Results/Simulation_Result_" + Type_Network + "_v%03d/" % version
#     maindir2 = stef_path
#     version += 200
# else:


if 'Darwin' in platform.system():
    data_dir = './'
else:
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

if not os.path.isdir("./Results"):
    os.mkdir("./Results")
maindir1 = "./Results/Simulation_Result_" + Type_Network + "_v%03d/" % version

if not os.path.isdir(maindir1):
    make_folder_results(maindir1)


logging.info('Create datasets')
# Data
if os.path.isfile(maindir1 + "All_Data_" + DataGroup + "_v%03d.pickle" % version):
    all_data = load(maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)
    train_dataset = all_data["Training data"]
    val_dataset = all_data["Validation data"]
    test_dataset = all_data["Testing data"]
else:
    test_dataset, val_dataset, train_dataset = Create_Datasets(
         data_dir+"Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False, channels=channels)
    all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
    save(all_data, maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)


# analyser
if os.path.isfile(maindir1 + Type_Network + "_analyser_v%03d.pickle" % version):
    analyser = load(maindir1 + Type_Network + "_analyser_v%03d" % version)
else:
    analyser = Analyser(maindir1)


# Model
if os.path.isfile(maindir1 + Type_Network + "_Project_v%03d.pt" % version):
    model = torch.load(maindir1 + Type_Network + "_Project_v%03d.pt" % version)
else:
    model = Network(device, channels)


# Learning Rate scheduler w. optimizer
if os.path.isfile(maindir1 + Type_Network + "_lrScheduler_v%03d.pickle" % version):
    scheduler_dict = load(maindir1 + Type_Network + "_lrScheduler_v%03d" % version)
    lrschedule = scheduler_dict["Type"]
    exp_lr_scheduler = scheduler_dict["Scheduler"]
else:
    # Optimizer
    optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # Add learning rate schedulers
    # Decay LR by a factor of gamma every step_size epochs
    lrschedule = 'plateau'
    if lrschedule == 'step':
        gamma = 0.5
        step_size = 40
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_algorithm, step_size=step_size, gamma=gamma)
    elif lrschedule == 'plateau':
        # Reduce learning rate when a metric has stopped improving
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)
        optimizer_algorithm = []
logging.info('Optimizer created')


# a = train_dataset[0]['image'][0:1,:,:]
# imshow(a)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)

root_dir = train_dataset.root_dir
img_path = train_dataset.All_Imagesets[0]
im_list = sorted(listdir(root_dir + img_path[1]))

model.to(device)

logging.info('Version %d' % version)
logging.info('Start training')
epochs=50
for epoch in range(epochs):
    epoch_start = time.time()

    logging.info('Epoch %d' % epoch)
    # for g in exp_lr_scheduler.optimizer.param_groups:
    """
    Here we can access analyser.validation_loss to make decisions
    """
    # Learning rate scheduler
    # perform scheduler step if independent from validation loss
    if lrschedule == 'step':
        exp_lr_scheduler.step()

    train(model, epoch, train_dataloader, val_dataloader, channels, device, plot=False,)
    # perform scheduler step if Dependent on validation loss
    if lrschedule == 'plateau':
        exp_lr_scheduler.step(analyser.validation_loss[-1])
    save_network(model, maindir1 + Type_Network + "_Project_v%03d" % version, device)
    torch.save(model, maindir1 + Type_Network + "_Project_v%03d.pt" % version)
    save(analyser, maindir1 + Type_Network + "_analyser_v%03d" % version)
    scheduler_dict = {"Type": lrschedule, "Scheduler": exp_lr_scheduler}
    save(scheduler_dict, maindir1 + Type_Network + "_lrScheduler_v%03d" % version)

    epoch_time = time.time() - epoch_start 
    logging.info('Epoch time: %.1f' % epoch_time)

# analyser = []
# model =[]
# exp_lr_scheduler = []
# scheduler_dict = []

# analyser.plot_loss()
# analyser.plot_accuracy()
# analyser.plot_loss_batchwise()
# analyser.plot_validation_loss()