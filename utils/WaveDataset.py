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
    transforms.Normalize(mean=[normalize['mean']], std=[normalize['std']])
]),
    "Train": transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[normalize['mean']], std=[normalize['std']])
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
    def __init__(self, root_directory, transform=None, check_bad_data=True, channels=3):
        self.root_dir = root_directory[0]
        self.classes = root_directory[1]
        self.imagesets = root_directory[2]
        self.transform = transform
        self.channels=channels


    def __len__(self):
        return len(self.imagesets)

    def __getitem__(self, idx):
#         logging.info('Get item')
        img_path = self.imagesets[idx][1]
        im_list = sorted(listdir(self.root_dir + img_path))

        Concat_Img = self.concatenate_data(img_path, im_list)

        sample = {"image": Concat_Img,
                  "target": torch.LongTensor([self.classes.index(str(img_path))])}
        return sample

    def concatenate_data(self, img_path, im_list):
        """
        Concatenated image tensor with all images having the same random transforms applied
        """
        for i, image in enumerate(im_list):
            if self.channels == 1:
                img = open_image(self.root_dir + img_path + "/" + image, grayscale=True)
            elif self.channels == 3:
                img = open_image(self.root_dir + img_path + "/" + image, grayscale=False)
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


def create_datasets(root_directory, transform=None, test_fraction=0., validation_fraction=0., check_bad_data=True, channels=3):
    """
    Splits data into fractional parts (data does not overlap!!) and creates data-loaders for each fraction.
    :param root_directory: Directory of data
    :param transform: transforms to apply for each data set. Must contain "Train" and "Test" dict
    :param test_fraction: Fraction of data to go to test-set
    :param validation_fraction: Fraction of data to go to validation-set
    :param check_bad_data: Option to evaluate and filter out corrupted data/images
    :return:
    """
    def filter_bad_data(img_path, channels):
        img = open_image(img_path, grayscale=True)
        good = False
        if (len(np.shape(img)) == 2) and (channels == 1):
            good = True
        elif (len(np.shape(img)) == 3) and (channels == np.shape(img)[-1]):
            good = True
        return good

    if (test_fraction > 0) or (validation_fraction > 0):
        bad_images = 0 
        good_images = 0
        classes = listdir(root_directory)
        imagesets = []
        for cla in classes:
            im_list = sorted(listdir(root_directory + cla))
            if not check_bad_data:
                imagesets.append((im_list, cla))
            else:
                Good = True
                for im in im_list:
                    Good = Good and filter_bad_data(root_directory + cla + "/" + im, channels)
                if Good:
                    imagesets.append((im_list, cla))
                    good_images += 1
                else:
                    bad_images += 1
        if check_bad_data:
            print('Loaded %d folders. Could not load %d folders' % (good_images, bad_images))

        full_size = len(imagesets)
        if test_fraction > 0:
            test = random.sample(imagesets, int(full_size * test_fraction)) # All images i list of t0s
            for item in test:
                imagesets.remove(item)

            Send = [root_directory, classes, test]
            Test = WaveDataset(Send, transform["Test"], channels=channels)
#             yield Test

        if validation_fraction > 0:
            validate = random.sample(imagesets, int(full_size * validation_fraction))  # All images i list of t0s
            for item in validate:
                imagesets.remove(item)

            Send = [root_directory, classes, validate]
            Validate = WaveDataset(Send, transform["Test"], channels=channels)
#             yield Validate

        Send = [root_directory, classes, imagesets]
        Train = WaveDataset(Send, transform["Train"], channels=channels)
#         yield Train
        return Test, Validate, Train
    else:
        Data = WaveDataset(root_directory, transform, check_bad_data=check_bad_data, channels=channels)
        return Data