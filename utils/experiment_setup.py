import platform
import logging
import torch
import os
from utils.io import save
from utils.WaveDataset import get_transforms, create_dataloaders, create_datasets


class ExperimentSetup():
    def __init__(self, args):
        self.args = args
        # self.dirs = self._get_dirs()

    def get_dirs(self):
        logging.info('Creating directories')
        if 'Darwin' in platform.system():
            dirs = {'base': '/Users/stathis/Code/thesis/wave_propagation/',
                    'data': '/Users/stathis/Code/thesis/wave_propagation/'}
        else:
            dirs = {'base': '/home/s1680171/wave_propagation/',
                    'data': '/disk/scratch/s1680171/wave_propagation/'}

        exp_folder = os.path.join(dirs['base'], "experiments_results/")
        if not os.path.isdir(exp_folder):
            os.mkdir(exp_folder)

        dirs['results'] = os.path.join(exp_folder, self.args.experiment_name)
        if not os.path.isdir(dirs['results']):
            os.mkdir(dirs['results'])

        for d in ['logs', 'pickles', 'models', 'predictions', 'charts']:
            dirs[d] = os.path.join(dirs['results'], '%s/' % d)
            if not os.path.isdir(dirs[d]):
                os.mkdir(dirs[d])
        self.dirs = dirs
        return dirs

    def get_dataloaders(self):
        # Datasets and data loaders
        transformations, normalizer = get_transforms(self.args.normalizer)
        logging.info('Creating datasets')
        train_dataset, val_dataset, test_dataset = create_datasets(os.path.join(self.dirs['data'], "Video_Data/"), transformations, test_fraction=0.15, validation_fraction=0.15)

        all_data = {"Training data": train_dataset,
                    "Validation data": val_dataset,
                    "Testing data": test_dataset}
        filename_data = os.path.join(self.dirs['results'], "all_data.pickle")
        save(all_data, filename_data)

        dataloaders = create_dataloaders(train_dataset, val_dataset, test_dataset, self.args.batch_size, self.args.num_workers)
        return dataloaders, normalizer

    def save_metadata(self, model, optim, lr_scheduler, device):
        filename_metadata = os.path.join(self.dirs['pickles'], "metadata.pickle")
        meta_data_dict = {"args": self.args, "optimizer": optim.state_dict(), "scheduler": lr_scheduler.state_dict(), "model": "%s" % model, 'device': device}
        save(meta_data_dict, filename_metadata)
        logging.info(meta_data_dict)

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info("use {} GPU(s)".format(torch.cuda.device_count()))
        else:
            logging.info("use CPU")
            device = torch.device('cpu')  # sets the device to be CPU
        return device