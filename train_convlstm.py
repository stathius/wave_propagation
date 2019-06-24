import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import platform
import logging
from collections import OrderedDict
from models.ConvLSTM import EncoderForecaster, Encoder, Forecaster, ConvLSTMCell
from utils.io import save_network, load_network, save, load, create_results_folder
from utils.arg_extract import get_args
from utils.ExperimentBuilder import ExperimentBuilder
from utils.WaveDataset import create_datasets, transformVar, normalize
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

results_dir = create_results_folder(base_folder=base_folder, experiment_name=args.experiment_name)
logging.info('Results dir: %s' % results_dir)
logging.info('Creating new datasets')
test_dataset, val_dataset, train_dataset = create_datasets(os.path.join(data_dir, "Video_Data/"), transformVar,
                                                           test_fraction=0.15, validation_fraction=0.15)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

filename_data = os.path.join(results_dir, "all_data.pickle")
all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
save(all_data, filename_data)


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

experiment = ExperimentBuilder(model=model, lr_scheduler=lr_scheduler,
                               experiment_name=args.experiment_name,
                               num_epochs=args.num_epochs,
                               device=device,
                               train_data=train_dataloader, val_data=val_dataloader, test_data=test_dataloader)
experiment_metrics, test_metrics = experiment.run_experiment()
