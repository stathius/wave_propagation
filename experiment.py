from __future__ import print_function
import logging
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import platform
import time
from utils.Network import Network
from utils.Analyser import Analyser
from utils.io import save_network, load_network, save, load
from utils.WaveDataset import create_datasets
from utils.training import train_epoch, validate, test

logging.basicConfig(format='%(message)s',level=logging.INFO)
channels=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformVar = {"Test": transforms.Compose([
    transforms.Resize(128),    #Already 184 x 184
    transforms.CenterCrop(128),
    transforms.ToTensor(),
]),
    "Train": transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    ])
}

nr_net = 0 

version = nr_net + 10
num_input_frames = 5
num_output_frames = 20
reinsert_frequency = 10
network_type = "7_kernel_3LSTM_debug"

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
    test_dataset, val_dataset, train_dataset = create_datasets(
         data_dir+"Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False, channels=channels)
    all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
    save(all_data, filename_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)


# analyser
filename_analyser = results_dir + "analyser.pickle" 
if os.path.isfile(filename_analyser):
    logging.info('Loading analyser')
    analyser = load(filename_analyser)
else:
    logging.info('Creating analyser')
    analyser = Analyser(results_dir)

# Model
filename_model = results_dir + "model.pt"
if os.path.isfile(filename_model):
    model = Network(device, channels)
    model = load_network(model, device, filename_model)
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
    epochs=50
    for epoch in range(epochs):
        epoch_start = time.time()

        logging.info('Epoch %d' % epoch)
        train_loss = train_epoch(model, lr_scheduler, epoch, train_dataloader, val_dataloader, num_input_frames, 
                                num_output_frames,reinsert_frequency, channels, device, analyser, plot=False)
        analyser.save_epoch_loss(train_loss, 1)
        validation_loss = validate(model, val_dataloader, num_input_frames, num_output_frames, reinsert_frequency, channels, device, plot=False)
        analyser.save_validation_loss(validation_loss, 1)
        """
        Here we can access analyser.validation_loss to make decisions
        """
        # Learning rate scheduler
        # perform scheduler step if independent from validation loss
        if scheduler_type == 'step':
            lr_scheduler.step()
        # perform scheduler step if dependent on validation loss
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