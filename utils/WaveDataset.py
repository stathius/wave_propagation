import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as FF
from os import listdir
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


def get_transforms(normalizer):
    normalizers = {'none': {'mean': 0.0, 'std': 1.0},  # leave as is
                   'normal': {'mean': 0.5047, 'std': 0.1176},  # mean 0 std 1
                   'm1to1': {'mean': 0.5, 'std': 0.5}  # makes it -1, 1
                   }
    normalizer = normalizers[normalizer]

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


def open_image(filename, grayscale=False):
    if grayscale:
        image = Image.open(filename).convert('L')
    else:
        image = Image.open(filename)
    return image


class WaveDataset(Dataset):
    """
    Creates a data-loader for the wave prop data
    """
    def __init__(self, data_directory, transform=None, check_bad_data=True):
        self.root_dir = data_directory[0]
        self.classes = data_directory[1]
        self.imagesets = data_directory[2]
        self.transform = transform

    def __len__(self):
        return len(self.imagesets)

    def __getitem__(self, idx):
        # logging.info('Get item')
        img_path = self.imagesets[idx][1]
        im_list = sorted(listdir(self.root_dir + img_path))

        Concat_Img = self.concatenate_data(img_path, im_list)

        return Concat_Img

    def concatenate_data(self, img_path, im_list):
        """
        Concatenated image tensor with all images having the same random transforms applied
        """
        for i, image in enumerate(im_list):
            img = open_image(self.root_dir + img_path + "/" + image, grayscale=True)

            if i == 0:
                if self.transform:
                    for t in self.transform.transforms:
                        if "RandomResizedCrop" in str(t):
                            ii, j, h, w = t.get_params(img, t.scale, t.ratio)
                            img = FF.resized_crop(img, ii, j, h, w, t.size, t.interpolation)
                        elif "RandomHorizontalFlip" in str(t):
                            Horizontal_Flip = random.choice([True, False])
                            if Horizontal_Flip:
                                img = FF.hflip(img)
                        elif "RandomVerticalFlip" in str(t):
                            Vertical_Flip = random.choice([True, False])
                            if Vertical_Flip:
                                img = FF.vflip(img)
                        else:
                            img = t(img)
                Concat_Img = img
            else:
                if self.transform:
                    for t in self.transform.transforms:
                        if "RandomResizedCrop" in str(t):
                            img = FF.resized_crop(img, ii, j, h, w, t.size, t.interpolation)
                        elif "RandomHorizontalFlip" in str(t):
                            if Horizontal_Flip:
                                img = FF.hflip(img)
                        elif "RandomVerticalFlip" in str(t):
                            if Vertical_Flip:
                                img = FF.vflip(img)
                        else:
                            img = t(img)
                Concat_Img = torch.cat((Concat_Img, img), dim=0)
        return Concat_Img


def create_datasets(data_directory, transform, test_fraction, validation_fraction):
    """
    Splits data into fractional parts (data does not overlap!!) and creates data-loaders for each fraction.
    :param data_directory: Directory of data
    :param transform: transforms to apply for each data set. Must contain "Train" and "Test" dict
    :param test_fraction: Fraction of data to go to test-set
    :param validation_fraction: Fraction of data to go to validation-set
    :param check_bad_data: Option to evaluate and filter out corrupted data/images
    :return:
    """
    classes = listdir(data_directory)
    imagesets = []
    for cla in classes:
        im_list = sorted(listdir(data_directory + cla))
        imagesets.append((im_list, cla))

    full_size = len(imagesets)

    test = random.sample(imagesets, int(full_size * test_fraction))  # All images i list of t0s
    for item in test:
        imagesets.remove(item)

    Send = [data_directory, classes, test]
    Test = WaveDataset(Send, transform["Test"])

    validate = random.sample(imagesets, int(full_size * validation_fraction))  # All images i list of t0s
    for item in validate:
        imagesets.remove(item)

    Send = [data_directory, classes, validate]
    Validate = WaveDataset(Send, transform["Test"])

    Send = [data_directory, classes, imagesets]
    Train = WaveDataset(Send, transform["Train"])

    return Train, Validate, Test


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloaders
