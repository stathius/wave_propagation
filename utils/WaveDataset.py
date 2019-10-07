import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as FF
import random
import os
from PIL import Image


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
    def __init__(self, data_directory, transform=None, back_and_forth=False):
        self.root_dir = data_directory[0]
        self.classes = data_directory[1]
        self.imagesets = data_directory[2]
        self.transform = transform
        self.back_and_forth = back_and_forth

    def __len__(self):
        return len(self.imagesets)

    def __getitem__(self, idx):
        # logging.info('Get item')
        img_path = self.imagesets[idx][1]
        # skip hidden files
        reverse = False
        if hasattr(self, 'back_and_forth'):
            if self.back_and_forth:
                reverse = bool(random.getrandbits(1))

        im_list = sorted([f for f in os.listdir(self.root_dir + img_path) if not f.startswith('.')], reverse=reverse)

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
