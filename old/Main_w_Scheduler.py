from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
import os
from os import listdir
import random
import copy
from torch.utils.data import Dataset, DataLoader
from skimage import measure #supports video also
from torchvision.transforms import functional as FF
import pickle
import scipy.ndimage as ndimage
import seaborn as sns
import pandas as pd
import imagehash
import math as m
from scipy.spatial import distance
import time

"""
Wilhelm Sorteberg, 2018
wilhelm@sorteberg.eu


"""


def hex_str2bool(hex_string):
    """
    Converts a hex string to boolean value (for pHash/wHash)
    :param hex_string: String of hex values
    :return: Boolean equivalent of the hex string
    """
    integerValue = 1
    for char in hex_string:
        integerValue *= 16
        if char == "0":
            integerValue += 0
        elif char == "1":
            integerValue += 1
        elif char == "2":
            integerValue += 2
        elif char == "3":
            integerValue += 3
        elif char == "4":
            integerValue += 4
        elif char == "5":
            integerValue += 5
        elif char == "6":
            integerValue += 6
        elif char == "7":
            integerValue += 7
        elif char == "8":
            integerValue += 8
        elif char == "9":
            integerValue += 9
        elif char == "a":
            integerValue += 10
        elif char == "b":
            integerValue += 11
        elif char == "c":
            integerValue += 12
        elif char == "d":
            integerValue += 13
        elif char == "e":
            integerValue += 14
        elif char == "f":
            integerValue += 15
    binary = bin(integerValue)
    output = []
    for i, element in enumerate(str(binary)):
        if i > 2:
            output.append(True if element == "1" else False)
    return np.asarray(output)




def imshow(inp, title=None, smoothen=False, return_np=False, obj=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if smoothen:
        inp = ndimage.gaussian_filter(inp, sigma=(.5, .5, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if obj is not None:
        obj.imshow(inp)
        obj.axis("off")
        if title is not None:
            obj.set_title(title)
    else:
        plt.imshow(inp)
        plt.axis("off")
        if title is not None:
            plt.title(title)
    if return_np:
        return inp

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformVar = {"Test": transforms.Compose([
    transforms.Resize(128),    #Already 184 x 184
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    normalize]),
    "Train": transforms.Compose([
    transforms.Resize(128),  # Already 184 x 184
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
    ])
}




"""
torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

    __len__ so that len(dataset) returns the size of the dataset.
    __getitem__ to support the indexing such that dataset[i] can be used to get i

th sample

"""


class Data_Analyser():
    """
    Saves network data for later analasys. Epochwise loss, Batchwise loss, Accuracy (not currently in use) and
    Validation loss
    """
    def __init__(self, maindir1):
        self.epoch_loss = []
        self.epoch_nr = []
        self.batch_loss = []
        self.batch_nr = []
        self.accuracy = []
        self.epoch_acc = []
        self.validation_loss = []
        self.validation_nr = []
        self.maindir1 = maindir1

    def save_loss(self, loss, epoch_increment=1):
        """
        Creates two lists, one of losses and one of index of epoch
        """
        self.epoch_loss.append(loss)
        self.epoch_nr.append(self.epoch_nr[len(self.epoch_nr) - 1] + epoch_increment) if len(self.epoch_nr)\
            else self.epoch_nr.append(epoch_increment)

    def plot_loss(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": self.epoch_nr, "Loss": self.epoch_loss})
        sns.lineplot(x="Epoch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.self.maindir1 + "Epoch_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_loss_batchwise(self, loss, batch_increment=1):
        """
        Creates two lists, one of losses and one of index of batch
        """
        self.batch_loss.append(loss)
        self.batch_nr.append(self.batch_nr[len(self.batch_nr) - 1] + batch_increment) if len(self.batch_nr)\
            else self.batch_nr.append(batch_increment)

    def plot_loss_batchwise(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Batch": self.batch_nr, "Loss": self.batch_loss})
        sns.lineplot(x="Batch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.self.maindir1 + "Batch_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_accuracy(self, accuracy, epoch_increment=1):
        """
        Creates two lists, one of accuracy and one of index of the accuracy (batchwise or epochwise)
        NOT IN USE
        """
        self.accuracy.append(accuracy)
        self.epoch_acc.append(self.epoch_acc[len(self.epoch_acc) - 1] + epoch_increment) if len(self.epoch_acc)\
            else self.epoch_acc.append(epoch_increment)

    def plot_accuracy(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": self.epoch_acc, "Accuracy": self.accuracy})
        sns.lineplot(x="Epoch", y="Accuracy",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.self.maindir1 + "Accuracy" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_validation_loss(self, loss, epoch_increment=1):
        """
        Creates two lists, one of validation losses and one of index of epoch
        """
        self.validation_loss.append(loss)
        self.validation_nr.append(self.validation_nr[len(self.validation_nr) - 1] + epoch_increment) if \
            len(self.validation_nr) else self.validation_nr.append(epoch_increment)

    def plot_validation_loss(self):
        """
        Plots validation and epoch loss next to each other
        """
        hue = []
        loss = []
        nr = []
        for i, element in enumerate(self.epoch_loss):
            loss.append(element)
            nr.append(self.epoch_nr[i])
            hue.append("Training Loss")
        for i, element in enumerate(self.validation_loss):
            loss.append(element)
            nr.append(self.validation_nr[i])
            hue.append("Validation Loss")
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": nr, "Loss": loss, "hue": hue})
        sns.lineplot(x="Epoch", y="Loss", hue="hue",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.maindir1 + "Validation_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()


"""
Saving and loading of figures, network state and other .pickle objects
"""
def save_network(obj, filename):
    network_dict = obj.cpu().state_dict()
    obj.cuda()
    save(obj=network_dict, filename=filename)

def save(obj, filename):
    filename += ".pickle" if ".pickle" not in filename else ""
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    filename += ".pickle" if ".pickle" not in filename else ""
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def figure_save(destination, obj=None):
    plt.savefig(destination)
    plt.savefig(destination + ".svg", format="svg")
    save(obj, destination) if obj else None

class Get_Data(Dataset):
    """
    Creates a data-loader.
    """
    def __init__(self, root_directory, transform=None, check_bad_data=True):
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


    def __len__(self):
        return len(self.All_Imagesets)

    def __getitem__(self, idx):
        img_path = self.All_Imagesets[idx]
        im_list = sorted(listdir(self.root_dir + img_path[1]))

        Concat_Img = self.concatenate_data(img_path, im_list)

        sample = {"image": Concat_Img,
                  "target": torch.LongTensor([self.classes.index(str(img_path[1]))])}
        return sample

    def filter_bad_data(self, img_path):
        img = Image.open(img_path)
        return False if np.shape(img)[-1] != 3 else True

    def concatenate_data(self, img_path, im_list):
        """
        Concatenated image tensor with all images having the same random transforms applied
        """
        for i, image in enumerate(im_list):
            if i == 0:
                img = Image.open(self.root_dir + img_path[1] + "/" + image)
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
                img = Image.open(self.root_dir + img_path[1] + "/" + image)
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


def Create_Training_Testing_Datasets(root_directory, transform=None, test_fraction=0., validation_fraction=0., check_bad_data=True):
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
        return False if np.shape(img)[-1] != 3 else True

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
            Test = Get_Data(Send, transform["Test"])
            yield Test

        if validation_fraction > 0:
            validate = random.sample(All_Imagesets, int(full_size * validation_fraction))  # All images i list of t0s
            for item in validate:
                All_Imagesets.remove(item)

            Send = [root_directory, classes, validate]
            Validate = Get_Data(Send, transform["Test"])
            yield Validate

        Send = [root_directory, classes, All_Imagesets]
        Train = Get_Data(Send, transform["Train"])
        yield Train

    else:
        Data = Get_Data(root_directory, transform, check_bad_data=check_bad_data)
        return Data

class Network (nn.Module):
    """
    The network structure
    """
    def __init__(self):
        super(Network, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(15, 60, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Conv2d(60, 120, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.Conv2d(120, 240, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.Conv2d(240, 480, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=480),
            nn.Tanh(),
            nn.Dropout2d(0.25)
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(30720, 1000),
            nn.Tanh(),
            nn.Dropout(0.25)
        )

        self.decoder_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1000, 30720)
        )

        self.decoder_conv = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose2d(480, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.ConvTranspose2d(240, 120, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.ConvTranspose2d(120, 60, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(60, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_0 = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)

        self.LSTM = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)

        self.LSTM_new_input = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)


    def forward(self, x, mode="input", training=False): #"input", "new_input", "internal"
        x.requires_grad_(training)
        with torch.set_grad_enabled(training):
            if "input" in mode:
                x = self.encoder_conv(x)
                self.org_size = x.size()
                x = x.view(-1, 30720)
                x = self.encoder_linear(x)
                if mode == "input":
                    self.h0, self.c0 = self.LSTM_0(x, (self.h0, self.c0))
                elif mode == "new_input":
                    self.h0, self.c0 = self.LSTM_new_input(x, (self.h0, self.c0))
            elif mode == "internal":
                self.h0, self.c0 = self.LSTM(self.h0, (self.h0, self.c0))
            x = self.h0.clone()
            x = self.decoder_linear(x)
            x = x.view(self.org_size)
            x = self.decoder_conv(x)
            return x

    def reset_hidden(self, batch_size, training=False):
        self.h0 = torch.zeros((batch_size, 1000), requires_grad=training).cuda() #Requires grad replaces Variable
        self.c0 = torch.zeros((batch_size, 1000), requires_grad=training).cuda()



def freeze_layers_old_model(old_model):
    for param in old_model.parameters():
        param.requires_grad = False

def stack_series(images):
    """
    If you want to stack images in channel direction
    """
    n = 2
    diff = images[:, (n - 1) * 3:n * 3, :, :] - images[:, n * 3:(n + 1) * 3, :, :]
    Stacked = torch.cat((images[:, n * 3:(n + 1) * 3, :, :], diff), dim=1)
    diff = images[:, (n + 1) * 3:(n + 2) * 3, :, :] - images[:, n * 3:(n + 1) * 3, :, :]
    Stacked = torch.cat((Stacked, diff), dim=1)
    diff = images[:, (n - 2) * 3:(n - 1) * 3, :, :] - images[:, n * 3:(n + 1) * 3, :, :]
    Stacked = torch.cat((Stacked, diff), dim=1)
    diff = images[:, (n + 2) * 3:(n + 3) * 3, :, :] - images[:, n * 3:(n + 1) * 3, :, :]
    Stacked = torch.cat((Stacked, diff), dim=1)
    return Stacked

class Scorekeeper():
    """
    Calculates and keeps track of testing results
    SSIM/pHash/RMSE etc.
    """
    def __init__(self):
        super(Scorekeeper, self).__init__()

        self.intermitted = []
        self.frame = []
        self.hue = []

        self.pHash_val = []
        self.pHash_frame = []
        self.pHash_hue = []

        self.pHash2_val = []
        self.pHash2_frame = []
        self.pHash2_hue = []

        self.SSIM_val = []
        self.SSIM_frame = []
        self.SSIM_hue = []

        self.MSE_val = []
        self.MSE_frame = []
        self.MSE_hue = []

        self.own = False
        self.phash = False
        self.SSIM = False
        self.MSE = False

    def add(self, predicted, target, frame_nr, *args):
        predicted = self.normalize(predicted)
        target = self.normalize(target)

        if "Own"in args:
            spatial_score, scale_score = self.score(predicted, target)
            self.intermitted.append(spatial_score)
            self.frame.append(frame_nr)
            self.hue.append("Spatial")
            self.intermitted.append(scale_score)
            self.frame.append(frame_nr)
            self.hue.append("Scaling")
            self.own = True

        if "SSIM" in args:
            ssim_score = self.ssim(predicted, target)
            self.SSIM_val.append(ssim_score)
            self.SSIM_frame.append(frame_nr)
            self.SSIM_hue.append("SSIM")
            self.SSIM = True

        if "RMSE" in args:
            self.MSE_val.append(np.sqrt(measure.compare_mse(predicted, target)))
            self.MSE_frame.append(frame_nr)
            self.MSE_hue.append("RMSE")
            self.MSE = True

        if "pHash" in args:
            hamming = self.pHash(predicted, target, "hamming")
            self.pHash_val.append(hamming)
            self.pHash_frame.append(frame_nr)
            self.pHash_hue.append("pHash")
            self.phash = True

        if "pHash" in args:
            hamming = self.pHash(predicted, target, "jaccard")
            self.pHash2_val.append(hamming)
            self.pHash2_frame.append(frame_nr)
            self.pHash2_hue.append("pHash")
            self.phash2 = True

    def hamming2(self, s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def pHash(self, predicted, target, *args):
        predicted = predicted * 255
        target = target * 255
        predicted = Image.fromarray(predicted.astype("uint8"))
        target = Image.fromarray(target.astype("uint8"))
        hash1 = hex_str2bool(str(imagehash.phash(predicted, hash_size=16)))
        hash2 = hex_str2bool(str(imagehash.phash(target, hash_size=16)))
        if "hamming" in args:
            return self.hamming2(hash1, hash2)
        elif "jaccard" in args:
            return distance.jaccard(hash1, hash2)
        else:
            return None
        # Out of 260

    def ssim(self, predicted, target):
        return measure.compare_ssim(predicted, target, multichannel=True, gaussian_weights=True)

    def normalize(self, image):
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        return np.clip(image, 0, 1)

    def score(self, predicted, target):
        predicted_mean = np.mean(predicted, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        pred_relative = np.abs(predicted - predicted_mean)
        target_relative = np.abs(target - target_mean)

        relative_diff = np.mean(np.abs(pred_relative - target_relative)) \
                        / (np.sum(target_relative) / np.prod(np.shape(target)))

        absolute_diff = np.mean(np.abs(predicted - target)) / (np.sum(target) / np.prod(np.shape(target)))

        return relative_diff, absolute_diff

    def plot(self):
        if self.own:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.frame, "Difference": self.intermitted, "Scoring Type": self.hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Difference", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig)
            figure_save(maindir1 + "Scoring_Quality", obj=fig)
            plt.show()

        if self.SSIM:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.SSIM_frame, "Similarity": self.SSIM_val,
                                  "Scoring Type": self.SSIM_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Similarity", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig)
            plt.ylim(0, 1)
            figure_save(maindir1 + "SSIM_Quality", obj=fig)
            plt.show()

        if self.MSE:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.MSE_frame, "Root Mean Square Error (L2 residual)": self.MSE_val,
                                 "Scoring Type": self.MSE_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Root Mean Square Error (L2 residual)", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig)
            figure_save(maindir1 + "RMSE_Quality", obj=fig)
            plt.show()

        if self.phash:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.pHash_frame, "Hamming Distance": self.pHash_val,
                                   "Scoring Type": self.pHash_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Hamming Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig)
            figure_save(maindir1 + "Scoring_Spatial_Hamming", obj=fig)
            plt.show()

        if self.phash2:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.pHash2_frame, "Jaccard Distance": self.pHash2_val,
                                   "Scoring Type": self.pHash2_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Jaccard Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig)
            figure_save(maindir1 + "Scoring_Spatial_Jaccard", obj=fig)
            plt.show()


def train(epoch, DataLoader, Validate, plot=True):
    """
    Training of the network
    :param epoch: Which epoch are you on
    :param DataLoader: Training data
    :param Validate: Validation data
    :return:
    """
    def initial_input(training):
        Data = ImageSeries[:, (t0 + n) * 3:(t0 + n + input_frames) * 3, :, :].cuda()
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()
        return output, target

    def new_input(output, target, training):
        output = torch.cat((output, model(
            output[:, -input_frames * 3:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()), dim=1
        )
        return output, target

    def consequent_propagation(output, target, training):
        output = torch.cat((output, model(torch.Tensor([0]), mode="internal", training=training)), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()), dim=1
        )
        return output, target

    def plot_predictions():
        if (i == 0) & (batch_num == 0):
            predicted = output[i, -3:, :, :].cpu().detach()
            des_target = target[i, -3:, :, :].cpu().detach()
            fig = plt.figure()
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % n, obj=tar)
            plt.show()

    print('Training Epoch: %d' % epoch)
    model.train()           # initialises training stage/functions
    mean_loss = 0
    print('Ready to load batches')
    for batch_num, batch in enumerate(DataLoader):
        print('Batch: %d' %batch_num)
        mean_batch_loss = 0
        Starting_times = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        for i, t0 in enumerate(Starting_times):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=True)
            exp_lr_scheduler.optimizer.zero_grad()
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(training=True)
                elif n == output_frames:
                    output, target = new_input(output, target, training=True)
                else:
                    output, target = consequent_propagation(output, target, training=True)
                if plot:
                    plot_predictions()
            loss = F.mse_loss(output, target)
            loss.backward()
            exp_lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()

        Analyser.save_loss_batchwise(mean_batch_loss / (i + 1), 1)
        mean_loss += loss.item()

        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_num + 1,
                   len(DataLoader), 100. * (batch_num + 1) / len(DataLoader), loss.item()))
    Analyser.save_loss(mean_loss / (batch_num + 1), 1)
    #Analyser.plot_loss()
    #Analyser.plot_loss_batchwise()
    validation_loss = validate(Validate, plot=False)
    Analyser.save_validation_loss(validation_loss, 1)
    Analyser.plot_validation_loss()
    print("Validation loss is", validation_loss)


def validate(DataLoader, plot=True):
    """
    Validation of network (same protocol as training)
    :param DataLoader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    def initial_input(training):
        Data = ImageSeries[:, (t0 + n) * 3:(t0 + n + input_frames) * 3, :, :].cuda()
        output = model(Data, training=training)
        target = ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()
        return output, target

    def new_input(output, target, training):
        output = torch.cat((output, model(
            output[:, -input_frames * 3:, :, :].clone(), mode="new_input", training=training)
                            ), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()), dim=1
        )
        return output, target

    def consequent_propagation(output, target, training):
        output = torch.cat((output, model(torch.Tensor([0]), mode="internal", training=training)), dim=1)
        target = torch.cat(
            (target, ImageSeries[:, (t0 + n + input_frames) * 3:(t0 + n + input_frames + 1) * 3, :, :].cuda()), dim=1
        )
        return output, target

    def plot_predictions():
        if (i == 0) & (batch_num == 0):
            predicted = output[i, -3:, :, :].cpu().detach()
            des_target = target[i, -3:, :, :].cpu().detach()
            fig = plt.figure()
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % n, obj=tar)
            plt.show()

    model.eval()
    overall_loss = 0
    for batch_num, batch in enumerate(DataLoader):
        Starting_times = random.sample(range(100 - input_frames - (2 * output_frames) - 1), 10)
        ImageSeries = batch["image"]
        batch_loss = 0
        for i, t0 in enumerate(Starting_times):
            model.reset_hidden(batch_size=ImageSeries.size()[0], training=False)
            for n in range(2 * output_frames):
                if n == 0:
                    output, target = initial_input(training=False)
                elif n == output_frames:
                    output, target = new_input(output, target, training=False)
                else:
                    output, target = consequent_propagation(output, target, training=False)
                if plot:
                    plot_predictions()
            batch_loss += F.mse_loss(output, target).item()
        overall_loss += batch_loss / (i + 1)
        print(batch_num + 1, "out of", len(DataLoader))
    return overall_loss / (batch_num + 1)


def test(DataLoader, plot=True):
    """
    Testing of network
    :param DataLoader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    def Smoothen(batch):
        """
        If you want to smoothen (Gaussian) the output images
        NOT USED
        """
        tran = transforms.ToTensor()
        for i in range(batch.size()[0]):
            if i == 0:
                inb = batch[i, :, :, :].numpy().transpose((1, 2, 0))
                inb = ndimage.gaussian_filter(inb, sigma=(1, 1, 0))
                out = tran(inb).unsqueeze_(0)
            else:
                inb = batch[i, :, :, :].numpy().transpose((1, 2, 0))
                inb = ndimage.gaussian_filter(inb, sigma=(1, 1, 0))
                inb = tran(inb).unsqueeze_(0)
                out = torch.cat((out, inb), dim=0)
        return out

    def initial_input(No_more_Target):
        output = model(ImageSeries.cuda())
        try:
            target = OriginalSeries[:, (t0 + cnt + input_frames) * 3:(t0 + cnt + input_frames + 1) * 3, :, :].cuda()
        except:
            No_more_Target = True
            target = None
        return output, target, No_more_Target

    def new_input(output, target, No_more_Target):
        output = torch.cat((output, model(ImageSeries, mode="new_input")), dim=1)
        try:
            target = torch.cat(
                (target, OriginalSeries[:, (t0 + cnt + input_frames) * 3:(t0 + cnt + input_frames + 1) * 3, :, :].cuda()
                 ), dim=1)
        except:
            No_more_Target = True
        return output, target, No_more_Target

    def consequent_propagation(output, target, No_more_Target):
        if n < (output_frames - refeed_offset):
            output = torch.cat((output, model(torch.Tensor([0]), mode="internal")), dim=1)
            try:
                target = torch.cat(
                    (target, OriginalSeries[:, (t0 + cnt + input_frames) * 3:(t0 + cnt + input_frames + 1) * 3, :, :].cuda()
                ), dim=1)
            except:
                No_more_Target = True
        return output, target, No_more_Target

    def plot_predictions():
        if (total == 0) & (n == 0) & (run == 0):
            for imag in range(int(ImageSeries.shape[1] / 3)):
                fig = plt.figure().add_axes()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                sns.set_context("talk")
                imshow(ImageSeries[selected_batch, imag * 3:(imag + 1) * 3, :, :], title="Input %01d" % imag, obj=fig)
                figure_save(maindir1 + "Input %02d" % imag)
        if (total == 0) & (n < (output_frames - refeed_offset)):
            predicted = output[selected_batch, -3:, :, :].cpu()
            des_target = target[selected_batch, -3:, :, :].cpu()
            fig = plt.figure()
            sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
            sns.set_context("talk")
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted %02d" % cnt, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % target_cnt, obj=tar)
            figure_save(maindir1 + "Prediction %02d" % cnt)
            plt.show() if plot else plt.close()

    def plot_cutthrough(frequently_plot=5, direction="Horizontal", location=None):
        def cutthrough(img1, img2,  hue1, hue2):
            intensity = []
            location = []
            hue = []
            if "orizontal" in direction:
                for i in range(np.shape(img1)[1]):
                    intensity.append(img1[stdmax[0], i, 0])
                    location.append(i)
                    hue.append(hue1)
                for i in range(np.shape(img2)[1]):
                    intensity.append(img2[stdmax[0], i, 0])
                    location.append(i)
                    hue.append(hue2)
            elif "ertical" in direction:
                for i in range(np.shape(img1)[0]):
                    intensity.append(img1[i, stdmax[0], 0])
                    location.append(i)
                    hue.append(hue1)
                for i in range(np.shape(img2)[0]):
                    intensity.append(img2[i, stdmax[0], 0])
                    location.append(i)
                    hue.append(hue2)

            data_dict = {"Intensity": intensity, "Pixel Location": location, "Image": hue}
            #g = sns.FacetGrid(pd.DataFrame.from_dict(data_dict), col="Image")
            #g.map(sns.lineplot, "Pixel Location", "Intensity")
            sns.lineplot(x="Pixel Location", y="Intensity", hue="Image",
                         data=pd.DataFrame.from_dict(data_dict), ax=profile)
            profile.set_title("Intensity Profile")

        if total == 0:
            if ((cnt + 1) % frequently_plot) == 0 or (cnt == 0):
                predicted = output[selected_batch, -3:, :, :].cpu()
                des_target = target[selected_batch, -3:, :, :].cpu()
                fig = plt.figure()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                with sns.axes_style("white"):
                    pre = fig.add_subplot(2, 2, 1)
                    tar = fig.add_subplot(2, 2, 2)
                with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
                    profile = fig.add_subplot(2, 2, (3, 4))

                predicted = imshow(predicted, title="Predicted %02d" % cnt, return_np=True, obj=pre)
                des_target = imshow(des_target, title="Target %02d" % target_cnt, return_np=True, obj=tar)
                if not location:
                    if "orizontal" in direction:
                        std = np.std(des_target, axis=1)
                    elif "ertical" in direction:
                        std = np.std(des_target, axis=0)
                    stdmax, _ = np.where(std.max() == std)
                else:
                    stdmax = location

                if "orizontal" in direction:
                    pre.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
                    tar.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
                elif "ertical" in direction:
                    pre.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")
                    tar.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")

                cutthrough(predicted, des_target, "Predicted", "Target")
                figure_save(maindir1 + "Cut-through %02d" % cnt, obj=fig)
                plt.show() if plot else plt.close()

    def add_score():
        if (not No_more_Target) & (n < (output_frames - refeed_offset)):
            for ba in range(output.size()[0]):
                score_keeper.add(output[ba, -3:, :, :].cpu(), target[ba, -3:, :, :].cpu(), cnt,
                                 "pHash", "SSIM", "Own", "RMSE")

    def introduce(prev_data):
        """
        If you want to introduce new droplets live during simulation
        NOT USED
        """
        def find_mean(input_img):
            for k in range(int(input_img.size()[0])):
                mean, number = np.unique(input_img[k:k + 1, :, :], return_counts=True)
                mean = np.full(np.shape(input_img[k:k + 1, :, :]), mean[np.argmax(number)])
                mean = torch.Tensor([mean])
                if k == 0:
                    matrix = mean
                else:
                    matrix = torch.cat((matrix, mean), dim=1)
            return matrix.squeeze_(0)

        prev_data = prev_data.cpu()
        data = My_Test[0]["image"][t0 * 3: (t0 + input_frames) * 3, :, :]
        for i in range(int(data.size()[0] / 3)):
            means = find_mean(data[i * 3:(i + 1) * 3, :, :])
            prev_data[selected_batch, i * 3:(i + 1) * 3, :, :] += data[i * 3:(i + 1) * 3, :, :] - means
        return prev_data


    model.eval()
    correct = total = 0
    t0 = 15 # Can be 0
    refeed_offset = 0
    selected_batch = random.randint(0, 15)
    if (output_frames - refeed_offset) < input_frames:
        refeed_offset = output_frames - input_frames
    for batch_num, batch in enumerate(DataLoader):
        OriginalSeries = batch["image"]
        ImageSeries = OriginalSeries[:, t0 * 3:(t0 + input_frames) * 3, :, :]
        #ImageSeries = introduce(ImageSeries)
        model.reset_hidden(ImageSeries.size()[0])
        No_more_Target = False
        cnt = target_cnt = 0
        for run in range(int(m.ceil((100 - (t0 + input_frames + 1)) / (output_frames - refeed_offset)))):
            if run != 0:
                if (refeed_offset == 0) or ((output_frames - refeed_offset) <= input_frames):
                    ImageSeries = output[:, -input_frames * 3:, :, :]
                else:
                    ImageSeries = output[:, -(input_frames + refeed_offset) * 3:-refeed_offset * 3, :, :]
                cnt -= refeed_offset
            for n in range(output_frames):
                if n == 0:
                    if run == 0:
                        output, target, No_more_Target = initial_input(No_more_Target)
                    else:
                        output, target, No_more_Target = new_input(output, target, No_more_Target)
                else:
                    output, target, No_more_Target = consequent_propagation(output, target, No_more_Target)
                    # output & target size is [batches, 3 * (n + 1), 128, 128]

                add_score()
                plot_predictions()
                plot_cutthrough()
                cnt += 1
                if not No_more_Target:
                    target_cnt = copy.copy(cnt)

        total += target.size()[0]
        print(batch_num + 1, "out of", len(DataLoader))
    print("Correct: {}\tPercentile: {:.0f}%".format(correct, 100 * correct / total))
    score_keeper.plot()

def make_folder_results(folder_name):
    if os.path.isdir(folder_name):
        imgs = os.listdir(folder_name)
        for img in imgs:
            os.remove(folder_name + "/" + img)
    else:
        os.mkdir(folder_name)

for nr_net in range(20):
    """
    Naming of network for saving
    """
    version = nr_net + 10
    input_frames = 5
    output_frames = 10
    Type_Network = "7_kernel_3LSTM"
    DataGroup = "LSTM"


    # Little trick to adjust path files for compatibility (I have a backup of the Main.py in case it doesn't work)
    # stef_path = "/media/sg6513/DATADRIVE2/MSc/Wavebox/"
    # if os.path.isfile(stef_path + "stefpc.txt"):
    #     if not os.path.isdir(stef_path + "Results"):
    #         os.mkdir(stef_path + "Results")
    #     maindir1 = stef_path + "Results/Simulation_Result_" + Type_Network + "_v%03d/" % version
    #     maindir2 = stef_path
    #     version += 200
    # else:
    if not os.path.isdir("./Results"):
        os.mkdir("./Results")
    # maindir1 = "/mnt/Linux-HDD/Discrete_Data-sharing_LSTMs/Results/Simulation_Result_"\
    maindir1 = "./Results/Simulation_Result_"\
               + Type_Network + "_v%03d/" % version
    maindir2 = "./"
    if not os.path.isdir(maindir1):
        make_folder_results(maindir1)

    # Data
    if os.path.isfile(maindir1 + "All_Data_" + DataGroup + "_v%03d.pickle" % version):
        My_Data = load(maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)
        My_Train = My_Data["Training data"]
        My_Validate = My_Data["Validation data"]
        My_Test = My_Data["Testing data"]
    else:
        My_Test, My_Validate, My_Train = Create_Training_Testing_Datasets(
            maindir2 + "Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False)
        My_Data = {"Training data": My_Train, "Validation data": My_Validate, "Testing data": My_Test}
        save(My_Data, maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)
    Train_Data = DataLoader(My_Train, batch_size=16, shuffle=True, num_workers=12)
    Validate_Data = DataLoader(My_Validate, batch_size=16, shuffle=True, num_workers=12)
    Test_Data = DataLoader(My_Test, batch_size=16, shuffle=True, num_workers=12)
    print('Data Loaded')

    # Analyser
    if os.path.isfile(maindir1 + Type_Network + "_Analyser_v%03d.pickle" % version):
        Analyser = load(maindir1 + Type_Network + "_Analyser_v%03d" % version)
    else:
        Analyser = Data_Analyser(maindir1)

    # Model
    if os.path.isfile(maindir1 + Type_Network + "_Project_v%03d.pt" % version):
        model = torch.load(maindir1 + Type_Network + "_Project_v%03d.pt" % version)
    else:
        model = Network()
    print('Network created')


    # Learning Rate scheduler w. optimizer
    if os.path.isfile(maindir1 + Type_Network + "_lrScheduler_v%03d.pickle" % version):
        scheduler_dict = load(maindir1 + Type_Network + "_lrScheduler_v%03d" % version)
        lrschedule = scheduler_dict["Type"]
        exp_lr_scheduler = scheduler_dict["Scheduler"]
    else:
        # Optimizer
        optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        # Add learning rate schedulers
        # Decay LR by a factor of gamma every step_size epochs
        lrschedule = 'plateau'
        if lrschedule == 'step':
            gamma = 0.5
            step_size = 40
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_algorithm, step_size=step_size, gamma=gamma)
        elif lrschedule == 'plateau':
            # Reduce learning rate when a metric has stopped improving
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)
            optimizer_algorithm = []
    print('Optimizer created')

    if torch.cuda.is_available():
        model.cuda()
    score_keeper = Scorekeeper()

    #Square_Data = DataLoader(My_Square, batch_size=16, shuffle=True, num_workers=12)
    #Line_Data = DataLoader(My_Line, batch_size=16, shuffle=True, num_workers=12)
    #Analyser.plot_loss()
    #Analyser.plot_accuracy()
    #Analyser.plot_loss_batchwise()
    # Analyser.plot_validation_loss()

    """
    Main Code
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _ in range(50 - len(Analyser.epoch_loss)):
        print('Version %d' % version)
        # for g in exp_lr_scheduler.optimizer.param_groups:
        """
        Here we can access Analyser.validation_loss to make decisions
        """
        # Learning rate scheduler
        # perform scheduler step if independent from validation loss
        if lrschedule == 'step':
            exp_lr_scheduler.step()
        train(len(Analyser.epoch_loss) + 1, Train_Data, Validate_Data, plot=False)
        # perform scheduler step if Dependent on validation loss
        if lrschedule == 'plateau':
            exp_lr_scheduler.step(Analyser.validation_loss[-1])
        save_network(model, maindir1 + Type_Network + "_Project_v%03d" % version)
        torch.save(model, maindir1 + Type_Network + "_Project_v%03d.pt" % version)
        save(Analyser, maindir1 + Type_Network + "_Analyser_v%03d" % version)
        scheduler_dict = {"Type": lrschedule, "Scheduler": exp_lr_scheduler}
        save(scheduler_dict, maindir1 + Type_Network + "_lrScheduler_v%03d" % version)
    test(Test_Data, plot=False)
    Analyser = []
    model =[]
    exp_lr_scheduler = []
    scheduler_dict = []
Analyser.plot_loss()
Analyser.plot_accuracy()
Analyser.plot_loss_batchwise()
Analyser.plot_validation_loss()
