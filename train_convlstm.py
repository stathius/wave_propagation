import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import platform
import logging
from collections import OrderedDict
from models.ConvLSTM import EncoderForecaster, Encoder, Forecaster, ConvLSTMCell
from utils.io import save_network, save, create_results_folder, save_datasets_to_file
from utils.arg_extract import get_args
from utils.ExperimentBuilder import ExperimentBuilder
from utils.WaveDataset import create_datasets, transformVar, normalize, create_dataloaders
plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args, device = get_args()  # get arguments from command line

# Data
if 'Darwin' in platform.system():
    base_folder = '/Users/stathis/Code/thesis/wave_propagation/'
    data_dir = base_folder
else:
    base_folder = '/home/s1680171/wave_propagation/'
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

dirs = create_results_folder(base_folder=base_folder, experiment_name=args.experiment_name)

logging.info('Creating datasets')
train_dataset, val_dataset, test_dataset = create_datasets(os.path.join(data_dir, "Video_Data/"), transformVar, test_fraction=0.15, validation_fraction=0.15)
filename_data = os.path.join(dirs['pickles'], "all_data.pickle")
save_datasets_to_file(train_dataset, val_dataset, test_dataset, filename_data)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers)


# Define encoder #
encoder_architecture = [
    [   # in_channels, out_channels, kernel_size, stride, padding
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTMCell(input_channel=8, num_filter=64, b_h_w=(args.batch_size, 64, 64),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_input_frames),
        ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 32, 32),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_input_frames),
        ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 16, 16),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_input_frames),
    ]
]
forecaster_architecture = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 4, 2, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 16, 16),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_output_frames),
        ConvLSTMCell(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 32, 32),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_output_frames),
        ConvLSTMCell(input_channel=64, num_filter=64, b_h_w=(args.batch_size, 64, 64),
                     kernel_size=3, stride=1, padding=1, device=device, seq_len=args.num_output_frames),
    ]
]

encoder = Encoder(encoder_architecture[0], encoder_architecture[1]).to(device)
forecaster = Forecaster(forecaster_architecture[0], forecaster_architecture[1], args.num_output_frames).to(device)
model = EncoderForecaster(encoder, forecaster)

optimizer = optim.Adam(model.parameters(), amsgrad=False, lr=args.learning_rate, weight_decay=args.weight_decay_coefficient)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

# Save metadata
filename_metadata = os.path.join(dirs['pickles'], "metadata.pickle")
meta_data_dict = {"args": args, "optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(), "model": "%s" % model}
save(meta_data_dict, filename_metadata)
logging.info(meta_data_dict)

experiment = ExperimentBuilder(model=model, lr_scheduler=lr_scheduler,
                               experiment_name=args.experiment_name,
                               num_epochs=args.num_epochs,
                               samples_per_sequence=args.samples_per_sequence,
                               device=device,
                               train_data=train_dataloader,
                               val_data=val_dataloader,
                               test_data=test_dataloader,
                               dirs=dirs,
                               continue_from_epoch=-1,
                               debug=args.debug)
experiment.run_experiment()
