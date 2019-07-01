import argparse
import torch
import random
import numpy as np
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, help='Network architecture for training [ar_lstm, convlstm, cnn]')
    parser.add_argument('--num_epochs', type=int, default=50, help='The experiment\'s epoch budget')
    parser.add_argument('--num_input_frames', type=int, default=5, help='LSTM. How many frames to insert initially')
    parser.add_argument('--num_output_frames', type=int, default=20, help='LSTM. How many framres to predict in the future"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--samples_per_sequence', type=int, default=10, help='how may training points to generate from a video sequence')
    parser.add_argument('--test_starting_point', type=int, default=15, help='which frame to start the test')
    parser.add_argument('--num_future_test_frames', type=int, default=80, help='how many frames to predict to the future')
    parser.add_argument('--reinsert_frequency', type=int, default=10, help='AR LSTM: how often to use the reinsert mechanism')
    parser.add_argument('--num_autoregress_frames', type=int, default=20, help='ConvLSTM: How many frames to keep from one pass to continue autoregression for longer outputs')
    parser.add_argument('--experiment_name', type=str, default="dummy",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--normalizer_type', type=str, default='normal', help='how to normalize the images [normal, m1to1, none]')
    parser.add_argument('--num_workers', type=int, default=12, help='how many workers for the dataloader')
    parser.add_argument('--seed', type=int, default=12345, help='Seed to use for random number generator for experiment')
    parser.add_argument('--seed_everything', type=str2bool, default=True)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--weight_decay_coefficient', type=float, default=1e-05, help='Weight decay to use for Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-03, help='learning rate to use for Adam')
    parser.add_argument('--scheduler_patience', type=int, default=7, help='Epoch patience before reducing learning_rate')
    parser.add_argument('--scheduler_factor', type=float, default=0.1, help='Factor to reduce learning_rate')
    parser.add_argument('--continue_experiment', type=str2bool, default=False, help='Whether the experiment should continue from the last epoch')

    args = parser.parse_args()

    if args.debug:
        args.num_input_frames = 2
        args.num_output_frames = 2
        args.batch_size = 2
        args.num_workers = 1
        args.samples_per_sequence = 5
        args.num_epochs = 3
        args.test_starting_point = 75
        args.normalizer_type = 'none'
        args.num_autoregress_frames = 5
        args.num_future_test_frames = 20

    if args.seed_everything:
        seed_everything(args.seed)

    return args
