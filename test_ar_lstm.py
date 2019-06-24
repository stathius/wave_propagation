from __future__ import print_function
import logging
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import sys
import platform
import time
from models.AR_LSTM import AR_LSTM, train_epoch, validate, test
from utils.Analyser import Analyser
from utils.io import save_network, load_network, save, load, create_results_folder
from utils.WaveDataset import create_datasets, transformVar
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
# import matplotlib
# matplotlib.use('Agg') # don't allow showing plots
import matplotlib.pyplot as plt
plt.ioff()

logging.basicConfig(format='%(message)s',level=logging.INFO)

if 'Darwin' in platform.system():
    base_folder = '/Users/stathis/Code/thesis/wave_propagation/'
    data_dir = base_folder
else:
    base_folder = '/home/s1680171/wave_propagation/'
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

experiment_name='ConvAE_LSTM_1ch_1'
# normalize = {'mean': 0.5047, 'std': 0.1176}
normalize = {'mean': 0.0, 'std': 1.0}

results_dir = base_folder + 'experiments_results/'+experiment_name

filename_data = os.path.join(results_dir,"all_data.pickle")

all_data = load(filename_data)
train_dataset = all_data["Training data"]
val_dataset = all_data["Validation data"]
test_dataset = all_data["Testing data"]
print(results_dir)

test_dataset.root_dir = data_dir + 'Video_Data/'
test_dataset.transform = transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[normalize['mean']], std=[normalize['std']])])

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12)

if torch.cuda.is_available():
    print('using gpu')
    device = torch.cuda.current_device()
else:
    device = 'cpu'

num_channels=1
model = AR_LSTM(num_channels=num_channels, device=device)
model = load_network(model, results_dir+"/model.pt")
model.to(device)

test_starting_point=15
num_input_frames=5
reinsert_frequency=10
show_plots=False
debug=False

# from utils.Scorekeeper import Scorekeeper
score_keeper=Scorekeeper(results_dir, normalize)
figures_dir = os.path.join(results_dir,'test_charts')
test(model, test_dataloader, test_starting_point, num_input_frames, reinsert_frequency,
            device, score_keeper, figures_dir, show_plots=show_plots, debug=debug, normalize=normalize)
score_keeper.plot(show_plots=show_plots)