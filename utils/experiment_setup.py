import platform
import logging
import torch
import os
import random
from utils.WaveDataset import WaveDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.io import save, load, save_json


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
    return trans


def create_new_datasets(data_directory, normalizer):
    logging.info('Creating new datasets')
    test_fraction = 0.15
    validation_fraction = 0.15
    transform = get_transforms(normalizer)

    classes = os.listdir(data_directory)
    imagesets = []
    for cla in classes:
        im_list = sorted(os.listdir(data_directory + cla))
        imagesets.append((im_list, cla))

    full_size = len(imagesets)

    test = random.sample(imagesets, int(full_size * test_fraction))  # All images i list of t0s
    for item in test:
        imagesets.remove(item)

    Send = [data_directory, classes, test]
    test_dataset = WaveDataset(Send, transform["Test"])

    validate = random.sample(imagesets, int(full_size * validation_fraction))  # All images i list of t0s
    for item in validate:
        imagesets.remove(item)

    Send = [data_directory, classes, validate]
    val_dataset = WaveDataset(Send, transform["Test"])

    Send = [data_directory, classes, imagesets]
    train_dataset = WaveDataset(Send, transform["Train"])

    datasets = {"Training data": train_dataset,
                "Validation data": val_dataset,
                "Testing data": test_dataset}
    return datasets


def load_datasets(filename_data):
    logging.info('Loading datasets')
    if os.path.isfile(filename_data):
        datasets = load(filename_data)
    else:
        raise Exception("Not a valid filename %s" % filename_data)
    return datasets


def create_dataloaders(datasets, batch_size, num_workers):
    train_dataset = datasets["Training data"]
    val_dataset = datasets["Validation data"]
    test_dataset = datasets["Testing data"]
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloaders


def save_metadata(filename_metadata, args, model, optim, lr_scheduler, device):
    meta_data_dict = {"args": args, "optimizer": optim.state_dict(), "scheduler": lr_scheduler.state_dict(), "model": "%s" % model, 'device': device}
    save_json(meta_data_dict, filename_metadata)
    save_json(meta_data_dict, filename_metadata + '.json')
    logging.info(meta_data_dict)


def load_metadata(filename_metadata):
    return load(filename_metadata)


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logging.info("use {} GPU(s)".format(torch.cuda.device_count()))
    else:
        logging.info("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
    return device


def save_network(model, filename):
    if hasattr(model, 'module'):
        network_dict = model.module.state_dict()
    else:
        network_dict = model.state_dict()
    torch.save(network_dict, filename)


def load_network(model, filename):
    dct = torch.load(filename, map_location='cpu')
    try:
        model.load_state_dict(dct)
    except Exception:
        raise Warning('model and dictionary mismatch')
    return model


def save_experiment():
    # TODO save and resume an experiment
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)


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
        self.files['datasets'] = os.path.join(self.dirs['pickles'], "datasets.pickle")
        self.files['metadata'] = os.path.join(self.dirs['pickles'], "metadata.pickle")
        self.files['analyser'] = os.path.join(self.dirs['pickles'], "analyser.pickle")
        self.files['model'] = os.path.join(self.dirs['models'], 'model.pt')

    def _create_dirs(self):
        logging.info('Creating directories')
        if not os.path.isdir(self.dirs['exp_folder']):
            os.mkdir(self.dirs['exp_folder'])
        if not os.path.isdir(self.dirs['results']):
            os.mkdir(self.dirs['results'])
        for d in self.sub_folders:
            if not os.path.isdir(self.dirs[d]):
                os.mkdir(self.dirs[d])

    def _get_dirs(self):
        if 'Darwin' in platform.system():
            dirs = {'base': '/Users/stathis/Code/thesis/wave_propagation/',
                    'data': '/Users/stathis/Code/thesis/wave_propagation/Video_Data/'}
        else:
            dirs = {'base': '/home/s1680171/wave_propagation/',
                    'data': '/disk/scratch/s1680171/wave_propagation/Video_Data/'}

        dirs['exp_folder'] = os.path.join(dirs['base'], "experiments_results/")
        dirs['results'] = os.path.join(dirs['exp_folder'], self.experiment_name)
        for d in self.sub_folders:
            dirs[d] = os.path.join(dirs['results'], '%s/' % d)

        return dirs
