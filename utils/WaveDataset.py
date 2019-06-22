import torch
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import functional as FF
from os import listdir
import numpy as np
import random
from PIL import Image
from torchvision import transforms

normalize = {'mean':0.5047, 'std':0.1176}

transformVar = {"Test": transforms.Compose([
    transforms.Resize(128),    #Already 184 x 184
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[normalize['mean']], std=[normalize['std']])
]),
    "Train": transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[normalize['mean']], std=[normalize['std']])
    ])
}

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
    def __init__(self, root_directory, transform=None, check_bad_data=True):
        self.root_dir = root_directory[0]
        self.classes = root_directory[1]
        self.imagesets = root_directory[2]
        self.transform = transform


    def __len__(self):
        return len(self.imagesets)

    def __getitem__(self, idx):
#         logging.info('Get item')
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


def create_datasets(root_directory, transform, test_fraction, validation_fraction):
    """
    Splits data into fractional parts (data does not overlap!!) and creates data-loaders for each fraction.
    :param root_directory: Directory of data
    :param transform: transforms to apply for each data set. Must contain "Train" and "Test" dict
    :param test_fraction: Fraction of data to go to test-set
    :param validation_fraction: Fraction of data to go to validation-set
    :param check_bad_data: Option to evaluate and filter out corrupted data/images
    :return:
    """
    classes = listdir(root_directory)
    imagesets = []
    for cla in classes:
        im_list = sorted(listdir(root_directory + cla))
        imagesets.append((im_list, cla))


    full_size = len(imagesets)

    test = random.sample(imagesets, int(full_size * test_fraction)) # All images i list of t0s
    for item in test:
        imagesets.remove(item)

    Send = [root_directory, classes, test]
    Test = WaveDataset(Send, transform["Test"])

    validate = random.sample(imagesets, int(full_size * validation_fraction))  # All images i list of t0s
    for item in validate:
        imagesets.remove(item)

    Send = [root_directory, classes, validate]
    Validate = WaveDataset(Send, transform["Test"])


    Send = [root_directory, classes, imagesets]
    Train = WaveDataset(Send, transform["Train"])

    return Test, Validate, Train