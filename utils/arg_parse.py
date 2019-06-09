import argparse
import os
import torch
import random
import numpy as np
import logging

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', nargs="?", type=int, default=12345, help='Seed to use for random number generator for experiment')
    # parser.add_argument('--model', type=str, help='Network architecture for training')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=50, help='The experiment\'s epoch budget')
    parser.add_argument('--num_input_frames', nargs="?", type=int, default=5)
    parser.add_argument('--num_output_frames', nargs="?", type=int, default=20)
    parser.add_argument('--reinsert_frequency', nargs="?", type=int, default=10)
    parser.add_argument('--num_channels', nargs="?", type=int, default=1, help='how many channels each frame has (gray/rgb)')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="dummy", 
                                            help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--seed_everything', type=str2bool, default=True)
    parser.add_argument('--plot', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()

    if args.seed_everything:
        seed_everything(args.seed)
 
    args.num_workers=12

    args.use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()
        logging.info("use {} GPU(s)".format(torch.cuda.device_count()))
    else:
        logging.info("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU

    return args, device