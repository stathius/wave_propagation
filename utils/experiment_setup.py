import platform
import logging
import torch
import os
from utils.WaveDataset import create_datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.io import save, load


def get_normalizer(normalizer):
    normalizers = {'none': {'mean': 0.0, 'std': 1.0},  # leave as is
                   'normal': {'mean': 0.5047, 'std': 0.1176},  # mean 0 std 1
                   'm1to1': {'mean': 0.5, 'std': 0.5}  # makes it -1, 1
                   }
    return normalizers[normalizer]


def get_transforms(normalizer):
    trans = {"Test": transforms.Compose([
        transforms.Resize(128),  # Already 184 x 184
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[normalizer['mean']], std=[normalizer['std']])
    ]), "Train": transforms.Compose([
        transforms.Resize(128),  # Already 184 x 184
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[normalizer['mean']], std=[normalizer['std']])])}
    return trans, normalizer


def create_new_datasets(data_dir, normalizer):
    logging.info('Creating new datasets')
    transformations, normalizer = get_transforms(normalizer)
    train_dataset, val_dataset, test_dataset = create_datasets(os.path.join(data_dir, "Video_Data/"), transformations, test_fraction=0.15, validation_fraction=0.15)

    datasets = {"Training data": train_dataset,
                "Validation data": val_dataset,
                "Testing data": test_dataset}
    return datasets


def load_datasets(self):
    logging.info('Loading datasets')
    filename_data = os.path.join(self.dirs['pickles'], self.filename_dataset)
    if os.path.isfile(filename_data):
        datasets = load(filename_data)
    else:
        raise Exception("Not a valid filename %s" % filename_data)
    return datasets


def create_dataloaders(datasets, batch_size, num_workers):
    train_dataset = datasets["Training data"]
    val_dataset = datasets["Validation data"]
    test_dataset = datasets["Testing data"]
    dataloaders= {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloaders


def save_metadata(filename_metadata, args, model, optim, lr_scheduler, device):
    meta_data_dict = {"args": args, "optimizer": optim.state_dict(), "scheduler": lr_scheduler.state_dict(), "model": "%s" % model, 'device': device}
    save(meta_data_dict, filename_metadata)
    logging.info(meta_data_dict)


def load_metadata(self):
    filename_metadata = os.path.join(self.dirs['pickles'], self.filename_metadata)
    metadata = load(filename_metadata)
    return metadata


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logging.info("use {} GPU(s)".format(torch.cuda.device_count()))
    else:
        logging.info("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
    return device


class ExperimentSetup():
    def __init__(self, experiment_name, new=True):
        logging.info('Experiment %s' % experiment_name)
        self.experiment_name = experiment_name
        self.sub_folders = ['logs', 'pickles', 'models', 'predictions', 'charts']
        self.new = new
        self.dirs = self._get_dirs()
        if self.new:
            self._create_dirs()
        self.files = {}
        self.files['dataset'] = os.path.join(self.dirs['pickles'], "datasets.pickle")
        self.files['metadata'] = os.path.join(self.dirs['pickles'], "metadata.pickle")
        self.files['analyser'] = os.path.join(self.dirs['pickles'], "analyser.pickle")
        self.files['model'] = os.path.join(self.dirs['models'], 'model.pt')

    def _create_dirs(self):
        if not os.path.isdir(self.dirs['exp_folder']):
            os.mkdir(self.dirs['exp_folder'])
        if not os.path.isdir(self.dirs['results']):
            os.mkdir(self.dirs['results'])
        for d in self.sub_folders:
            if not os.path.isdir(self.dirs[d]):
                os.mkdir(self.dirs[d])

    def _get_dirs(self):
        logging.info('Creating directories')
        if 'Darwin' in platform.system():
            dirs = {'base': '/Users/stathis/Code/thesis/wave_propagation/',
                    'data': '/Users/stathis/Code/thesis/wave_propagation/'}
        else:
            dirs = {'base': '/home/s1680171/wave_propagation/',
                    'data': '/disk/scratch/s1680171/wave_propagation/'}

        dirs['exp_folder'] = os.path.join(dirs['base'], "experiments_results/")
        dirs['results'] = os.path.join(dirs['exp_folder'], self.experiment_name)
        for d in self.sub_folders:
            dirs[d] = os.path.join(dirs['results'], '%s/' % d)
        return dirs









