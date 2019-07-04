import platform
import logging
import torch
import os
import random
from argparse import Namespace
from utils.WaveDataset import WaveDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.io import save, load, save_json, load_json
from models.AR_LSTM import AR_LSTM
from models.ConvLSTM import get_convlstm_model
from models.ResNet import resnet12
from utils.Logger import Logger


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


class Experiment():
    def __init__(self, args):
        logging.info('Experiment %s' % args.experiment_name)
        self.args = args
        self.sub_folders = ['pickles', 'models', 'predictions', 'charts', 'training']
        self._filesystem_structure()
        self.device = get_device()

    def _create_model(self, model_type):
        if model_type == 'convlstm':
            model = get_convlstm_model(self.args.num_input_frames, self.args.num_output_frames, self.args.batch_size, self.device)
        elif model_type == 'ar_lstm':
            model = AR_LSTM(self.args.num_input_frames, self.args.reinsert_frequency, self.device)
        elif model_type == 'resnet':
            model = resnet12(self.args.num_input_frames, self.args.num_output_frames)
        return model

    def _create_scheduler(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=self.args.scheduler_factor,
                                                          patience=self.args.scheduler_patience)

    def create_new(self):
        self._mkdirs()
        self.normalizer = get_normalizer(self.args.normalizer_type)
        self.datasets = create_new_datasets(self.dirs['data'], self.normalizer)
        save(self.datasets, self.files['datasets'])
        self.dataloaders = create_dataloaders(self.datasets, self.args.batch_size, self.args.num_workers)
        self.model = self._create_model(self.args.model_type)
        self.lr_scheduler = self._create_scheduler()
        self._save_metadata()
        self.logger = Logger()
        self.model.to(self.device)

    def load_from_disk(self, test=True):
        self.metadata = self._load_metadata()
        self.args_new = self.args
        self.args = Namespace(**self.metadata['args'])
        self.normalizer = get_normalizer(self.args.normalizer_type)
        self.datasets = load_datasets(self.files['datasets'])
        self.datasets['Training data'].root_dir = self.dirs['data']
        self.datasets['Training data'].transform = get_transforms(self.normalizer)['Train']
        self.datasets['Validation data'].root_dir = self.dirs['data']
        self.datasets['Validation data'].transform = get_transforms(self.normalizer)['Test']
        self.datasets['Testing data'].root_dir = self.dirs['data']
        self.datasets['Testing data'].transform = get_transforms(self.normalizer)['Test']
        self.dataloaders = create_dataloaders(self.datasets, self.args.batch_size, self.args.num_workers)
        self.model = self._create_model(self.args.model_type)
        self.lr_scheduler = self._create_scheduler()
        if test:
            self.model = load_network(self.model, self.files['model_best'])
        else:
            self.model = load_network(self.model, self.files['model_lastest'])
        self.model.to(self.device)
        # Plus more stuff to get the best val accuracy and the last epoch number

    def _save_metadata(self):
        meta_data_dict = {"args": self.args,
                          "optimizer": self.lr_scheduler.optimizer.state_dict(),
                          "scheduler": self.lr_scheduler.state_dict(),
                          "model": "%s" % self.model
                          }
        save(meta_data_dict, self.files['metadata'])
        save_json(meta_data_dict, self.files['metadata'] + '.json')
        logging.info(meta_data_dict)

    def _load_metadata(self):
        mtd = load_json(self.files['metadata'] + '.json')
        return mtd
        # return load(self.files['metadata'])

    def _mkdirs(self):
        logging.info('Creating directories')
        if not os.path.isdir(self.dirs['exp_folder']):
            os.mkdir(self.dirs['exp_folder'])
        if not os.path.isdir(self.dirs['results']):
            os.mkdir(self.dirs['results'])
        for d in self.sub_folders:
            if not os.path.isdir(self.dirs[d]):
                os.mkdir(self.dirs[d])

    def _filesystem_structure(self):
        self.dirs = {}
        if 'Darwin' in platform.system():
            self.dirs['base'] = '/Users/stathis/Code/thesis/wave_propagation/'
            self.dirs['data'] = '/Users/stathis/Code/thesis/wave_propagation/Video_Data/'
        else:
            self.dirs['base'] = '/home/s1680171/wave_propagation/'
            self.dirs['data'] = '/disk/scratch/s1680171/wave_propagation/Video_Data/'

        self.dirs['exp_folder'] = os.path.join(self.dirs['base'], "experiments_results/")
        self.dirs['results'] = os.path.join(self.dirs['exp_folder'], self.args.experiment_name)
        for d in self.sub_folders:
            self.dirs[d] = os.path.join(self.dirs['results'], '%s/' % d)

        self.files = {}
        self.files['datasets'] = os.path.join(self.dirs['pickles'], "datasets.pickle")
        self.files['metadata'] = os.path.join(self.dirs['pickles'], "metadata.pickle")
        self.files['logger'] = os.path.join(self.dirs['pickles'], "logger.json")
        self.files['evaluator'] = os.path.join(self.dirs['pickles'], 'evaluator_sp_%d.pickle')
        self.files['model_latest'] = os.path.join(self.dirs['models'], 'model_latest.pt')
        self.files['model_best'] = os.path.join(self.dirs['models'], 'model_best.pt')
        # self.files['progress'] = os.path.join(self.dirs['training'], "progress.json")

