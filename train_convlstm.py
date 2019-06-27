import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict
from models.ConvLSTM import EncoderForecaster, Encoder, Forecaster, ConvLSTMCell
from utils.arg_extract import get_args_train
from utils.experiment_runner import ExperimentRunner
from utils.experiment_setup import ExperimentSetup, get_normalizer, create_new_datasets, create_dataloaders, get_device, save_metadata
from utils.io import save

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args_train()
setup = ExperimentSetup(args.experiment_name)
normalizer = get_normalizer(args.normalizer)
datasets = create_new_datasets(setup.dirs['data'], normalizer)
save(datasets, setup.files['dataset'])
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()


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

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

save_metadata(setup.files['metadata'], args, model, optimizer, lr_scheduler, device)

experiment = ExperimentRunner(model=model, lr_scheduler=lr_scheduler,
                             experiment_name=args.experiment_name,
                             num_epochs=args.num_epochs,
                             samples_per_sequence=args.samples_per_sequence,
                             device=device,
                             train_data=data_loaders['train'],
                             val_data=data_loaders['val'],
                             test_data=data_loaders['test'],
                             dirs=setup.dirs,
                             continue_from_epoch=-1,
                             debug=args.debug)
experiment.run_experiment()
