import torch
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import functional as FF
from os import listdir
import random
from PIL import Image

class Get_Dataset(Dataset):
    """
    Creates a data-loader.
    """
    def __init__(self, root_directory, transform=None, check_bad_data=True, channels=3):
        if isinstance(root_directory, str):
            self.root_dir = root_directory
            self.classes = listdir(root_directory)
            self.All_Imagesets = []
            for cla in self.classes:
                im_list = sorted(listdir(root_directory + cla))
                if not check_bad_data:
                    self.All_Imagesets.append((im_list, cla))
                else:
                    Good = True
                    for im in im_list:
                        Good = Good and self.filter_bad_data(root_directory + cla + "/" + im)
                    if Good:
                        self.All_Imagesets.append((im_list, cla))

        elif isinstance(root_directory, list):
            self.root_dir = root_directory[0]
            self.classes = root_directory[1]
            self.All_Imagesets = root_directory[2]

        self.transform = transform
        self.channels=channels


    def __len__(self):
        return len(self.All_Imagesets)

    def __getitem__(self, idx):
#         print('Get item')
        img_path = self.All_Imagesets[idx][1]
        im_list = sorted(listdir(self.root_dir + img_path))

        Concat_Img = self.concatenate_data(img_path, im_list)

        sample = {"image": Concat_Img,
                  "target": torch.LongTensor([self.classes.index(str(img_path))])}
        return sample

    def filter_bad_data(self, img_path):
        img = Image.open(img_path)
        return False if np.shape(img)[-1] != channels else True

    def concatenate_data(self, img_path, im_list):
        """
        Concatenated image tensor with all images having the same random transforms applied
        """
        for i, image in enumerate(im_list):
            if self.channels == 1:
                img = Image.open(self.root_dir + img_path + "/" + image).convert('L')
            elif self.channels == 3:
                img = Image.open(self.root_dir + img_path + "/" + image)
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


def Create_Datasets(root_directory, transform=None, test_fraction=0., validation_fraction=0., check_bad_data=True, channels=3):
    """
    Splits data into fractional parts (data does not overlap!!) and creates data-loaders for each fraction.
    :param root_directory: Directory of data
    :param transform: transforms to apply for each data set. Must contain "Train" and "Test" dict
    :param test_fraction: Fraction of data to go to test-set
    :param validation_fraction: Fraction of data to go to validation-set
    :param check_bad_data: Option to evaluate and filter out corrupted data/images
    :return:
    """
    def filter_bad_data(img_path):
        img = Image.open(img_path)
        return False if np.shape(img)[-1] != channels else True

    print('Create datasets')
    if (test_fraction > 0) or (validation_fraction > 0):
        classes = listdir(root_directory)
        All_Imagesets = []
        for cla in classes:
            im_list = sorted(listdir(root_directory + cla))
            if not check_bad_data:
                All_Imagesets.append((im_list, cla))
            else:
                Good = True
                for im in im_list:
                    Good = Good and filter_bad_data(root_directory + cla + "/" + im)
                if Good:
                    All_Imagesets.append((im_list, cla))

        full_size = len(All_Imagesets)
        if test_fraction > 0:
            test = random.sample(All_Imagesets, int(full_size * test_fraction)) # All images i list of t0s
            for item in test:
                All_Imagesets.remove(item)

            Send = [root_directory, classes, test]
            Test = Get_Dataset(Send, transform["Test"], channels=channels)
#             yield Test

        if validation_fraction > 0:
            validate = random.sample(All_Imagesets, int(full_size * validation_fraction))  # All images i list of t0s
            for item in validate:
                All_Imagesets.remove(item)

            Send = [root_directory, classes, validate]
            Validate = Get_Dataset(Send, transform["Test"], channels=channels)
#             yield Validate

        Send = [root_directory, classes, All_Imagesets]
        Train = Get_Dataset(Send, transform["Train"], channels=channels)
#         yield Train
        return Test, Validate, Train
    else:
        Data = Get_Dataset(root_directory, transform, check_bad_data=check_bad_data, channels=channels)
        return Data