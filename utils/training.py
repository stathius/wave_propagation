import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.io import figure_save
import random
import math
import numpy as np
from PIL import Image
import copy
import logging
import time
import os
from utils.io import imshow
# from utils.WaveDataset import normalize

def initial_input(model, input_frames, batch_images, starting_point, num_input_frames, num_channels, device, training):
    """
    var           size
    batch_images  [16, 100, 128, 128]
    input_frames  [16, 5, 128, 128]
    output_frames  [16, 1, 128, 128]
    target_frames  [16, 1, 128, 128]
    """
    output_frames = model(input_frames.to(device), mode='initial_input', training=training)
    target_idx = starting_point + num_input_frames
    target_frames = batch_images[:, target_idx * num_channels:(target_idx + 1) * num_channels, :, :].to(device)
    return output_frames, target_frames

def reinsert(model, input_frames, output_frames, target_frames, batch_images, starting_point, num_input_frames, future_frame_idx, num_channels, device, training):
    output_frames = torch.cat((output_frames, model(input_frames, mode="reinsert", training=training)), dim=1).to(device)
    target_idx = starting_point + future_frame_idx + num_input_frames
    target_frames = torch.cat((target_frames, batch_images[:, target_idx*num_channels:(target_idx + 1) * num_channels, :, :].to(device)), dim=1)
    return output_frames, target_frames

def propagate(model, output_frames, target_frames, batch_images, starting_point, num_input_frames, future_frame_idx, num_channels, device, training):
    output_frames = torch.cat((output_frames, model(torch.Tensor([0]), mode="propagate", training=training)), dim=1).to(device)
    target_idx = starting_point + future_frame_idx + num_input_frames
    target_frames = torch.cat((target_frames, batch_images[:, target_idx* num_channels:(target_idx+ 1) * num_channels, :, :].to(device)), dim=1)
    return output_frames, target_frames

def plot_predictions(batch, output_frames, target_frames, num_channels, show_plots): 
    logging.info('** plot predictions **')
    predicted = output_frames[batch, -num_channels:, :, :].cpu().detach()
    des_target_frames = target_frames[batch, -num_channels:, :, :].cpu().detach()
    fig = plt.figure(figsize=[8,8])
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted smoothened %02d" % batch, smoothen=True, obj=pred, normalize=normalize)
    tar = fig.add_subplot(1, 2, 2)
    imshow(des_target_frames, title="Target %02d" % batch, obj=tar, normalize=normalize)
    if show_plots:
        plt.show()


def train_epoch(model, lr_scheduler, epoch, train_dataloader, num_input_frames, num_output_frames, 
                    reinsert_frequency, num_channels, device, analyser, show_plots=False, debug=False):
    """
    Training of the network
    :param train: Training data
    :return:
    """
    training = True
    model.train()           # initialises training stage/functions
    mean_loss = 0
    # logging.info('Training: Ready to load batches')
    for batch_num, batch in enumerate(train_dataloader):
        batch_start = time.time()
        # logging.info('Batch: %d loaded in %.3f' %(batch_num, batch_time))
        mean_batch_loss = 0
        random_starting_points = random.sample(range(100 - num_input_frames - num_output_frames - 1), 10)
        batch_images = batch["image"]
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=batch_images.size()[0], training=True)
            if training:
                lr_scheduler.optimizer.zero_grad()
            for future_frame_idx in range(num_output_frames):
                if future_frame_idx == 0:
                    # Take the first 5 frames of the batch starting from the random starting_point
                    input_frames = batch_images[:, starting_point * num_channels:(starting_point + num_input_frames) * num_channels, :, :].clone()
                    output_frames, target_frames = initial_input(model, input_frames, batch_images, starting_point, num_input_frames, num_channels, device, training) 
                elif future_frame_idx == reinsert_frequency:
                    # It will insert the last 5 predictions as an input
                    input_frames = output_frames[:, -num_input_frames * num_channels:, :, :].clone()
                    output_frames, target_frames = reinsert(model, input_frames, output_frames, target_frames, batch_images, 
                                                            starting_point, num_input_frames, future_frame_idx, num_channels, device, training)
                else:
                    # This doesn't take any input, just propagates the LSTM internal state once
                    output_frames, target_frames = propagate(model, output_frames, target_frames, batch_images, starting_point, 
                                                             num_input_frames, future_frame_idx, num_channels, device, training)
            loss = F.mse_loss(output_frames, target_frames)
            if training:
                loss.backward()
                lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()
            if debug: break

        analyser.save_loss_batchwise(mean_batch_loss / (i + 1), batch_increment=1)
        mean_loss += loss.item()

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1,
                   len(train_dataloader), 100. * (batch_num + 1) / len(train_dataloader), loss.item(), batch_time ) )        

        if debug: break
    epoch_loss = mean_loss / (batch_num + 1)
    # plot_predictions(batch_num, output_frames, target_frames, num_channels, show_plots)
    return epoch_loss


def validate(model, val_dataloader, num_input_frames, num_output_frames ,reinsert_frequency, num_channels, device, show_plots=False, debug=False):
    """
    Validation of network (same protocol as training)
    :param val_dataloader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    training = False
    model.eval()
    overall_loss = 0
    for batch_num, batch in enumerate(val_dataloader):
        random_starting_points = random.sample(range(100 - num_input_frames - num_output_frames - 1), 10)
        batch_images = batch["image"]
        batch_loss = 0
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=batch_images.size()[0], training=False)
            for future_frame_idx in range(num_output_frames):
                if future_frame_idx == 0:
                    input_frames = batch_images[:, starting_point * num_channels:(starting_point + num_input_frames) * num_channels, :, :].clone()
                    output_frames, target_frames = initial_input(model, input_frames, batch_images, starting_point, num_input_frames, num_channels, device, training) 
                elif future_frame_idx == reinsert_frequency:
                    input_frames = output_frames[:, -num_input_frames * num_channels:, :, :].clone()
                    output_frames, target_frames = reinsert(model, input_frames, output_frames, target_frames, batch_images, 
                                                            starting_point, num_input_frames, future_frame_idx, num_channels, device, training)
                else:
                    output_frames, target_frames = propagate(model, output_frames, target_frames, batch_images, starting_point, 
                                                             num_input_frames, future_frame_idx, num_channels, device, training)
            loss = F.mse_loss(output_frames, target_frames)
            batch_loss += loss.item()
            if debug: break
        overall_loss += batch_loss / (i + 1)
        if debug: break
    val_loss = overall_loss / (batch_num + 1)
    # plot_predictions(batch_num, output_frames, target_frames, num_channels, show_plots)
    return val_loss


def test(model, test_dataloader, starting_point, num_input_frames, reinsert_frequency, 
            num_channels, device, score_keeper, figures_dir, show_plots=False, debug=False, normalize=None):
    """
    Testing of network
    :param test_dataloader: Data to test
    :param plot: If to plot predictionss
    :return:
    """

    def plot_predictions(show_plots=False):
        if future_frame_idx == 0:
            for imag in range(int(input_frames.shape[1] / num_channels)):
                fig = plt.figure().add_axes()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                sns.set_context("talk")
                imshow(input_frames[image_to_plot, imag * num_channels:(imag + 1) * num_channels, :, :], title="Input %01d" % imag, obj=fig, normalize=normalize)
                figure_save(os.path.join(figures_dir,"Input %02d" % imag))
        predicted = output_frames[image_to_plot, -num_channels:, :, :].cpu()
        des_target = target_frames[image_to_plot, -num_channels:, :, :].cpu()
        fig = plt.figure()
        sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        pred = fig.add_subplot(1, 2, 1)
        imshow(predicted, title="Predicted %02d" % future_frame_idx, smoothen=True, obj=pred, normalize=normalize)
        tar = fig.add_subplot(1, 2, 2)
        imshow(des_target, title="Target %02d" % future_frame_idx, obj=tar, normalize=normalize)
        figure_save(os.path.join(figures_dir,"Prediction %02d" % future_frame_idx))
        if show_plots:
            plt.show()

    def plot_cutthrough(direction="Horizontal", location=None, show_plots=False):
        def cutthrough(img1, img2,  hue1, hue2):
            intensity = []
            location = []
            hue = []
            if "Horizontal" in direction:
                intensity = np.append(img1[stdmax[0],:], img2[stdmax[0],:])
                length1 = img1.shape[1]
                length2 = img2.shape[1]
                location = list(range(length1)) + list(range(length2))
                hue = [hue1] * length1 + [hue2] * length2
            elif "Vertical" in direction:
                intensity = np.append(img1[:,stdmax[0]], img2[:,stdmax[0]])
                width1 = img1.shape[0]
                width2 = img2.shape[0]
                location = list(range(width1)) + list(range(width2))
                hue = [hue1] * width1 + [hue2] * width2

            data_dict = {"Intensity": intensity, "Pixel Location": location, "Image": hue}
            sns.lineplot(x="Pixel Location", y="Intensity", hue="Image",
                         data=pd.DataFrame.from_dict(data_dict), ax=profile)
            profile.set_title("Intensity Profile")

        num_channels = 1 # override num_channels to plot correctly
        predicted = output_frames[image_to_plot, -num_channels:, :, :].cpu()
        des_target = target_frames[image_to_plot, -num_channels:, :, :].cpu()
        fig = plt.figure()
        sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
        with sns.axes_style("white"):
            pre = fig.add_subplot(2, 2, 1)
            tar = fig.add_subplot(2, 2, 2)
        with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
            profile = fig.add_subplot(2, 2, (3, 4))

        predicted = imshow(predicted, title="Predicted %02d" % future_frame_idx, return_np=True, obj=pre, normalize=normalize)
        des_target = imshow(des_target, title="Target %02d" % future_frame_idx, return_np=True, obj=tar, normalize=normalize)
        if not location:
            if "Horizontal" in direction:
                std = np.std(des_target, axis=1)
            elif "Vertical" in direction:
                std = np.std(des_target, axis=0)
            stdmax = np.where(std.max() == std)
        else:
            stdmax = location

        if "Horizontal" in direction:
            pre.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
            tar.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
        elif "Vertical" in direction:
            pre.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")
            tar.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")

        # print(predicted.size())
        cutthrough(predicted, des_target, "Predicted", "Target")
        figure_save(os.path.join(figures_dir, "Cut-through %02d" % future_frame_idx), obj=fig)
        if show_plots:
            plt.show()

    model.eval()
    total = 0
    image_to_plot = random.randint(0, 15)
    reinsert_frequency = 10
    training = False
    # plot_frequency = 5

    for batch_num, batch in enumerate(test_dataloader):
        batch_images = batch["image"]
        batch_size = batch_images.size()[0]
        model.reset_hidden(batch_size=batch_images.size()[0], training=False)
        
        total_frames = batch_images.size()[1]
        num_future_frames = total_frames - (starting_point + num_input_frames)
        for future_frame_idx in range(num_future_frames):
            if future_frame_idx == 0:
                prop_type = 'Initial input'
                input_frames = batch_images[:, starting_point * num_channels:(starting_point + num_input_frames) * num_channels, :, :].clone()
                output_frames, target_frames = initial_input(model, input_frames, batch_images, starting_point, num_input_frames, num_channels, device, training)
                prop_type = 'Reinsert'
                input_frames = output_frames[:, -num_input_frames * num_channels:, :, :].clone()
                output_frames, target_frames = reinsert(model, input_frames, output_frames, target_frames, batch_images, 
                                                                starting_point, num_input_frames, future_frame_idx, num_channels, device, training)
            else:
                prop_type = 'Propagate'
                output_frames, target_frames = propagate(model, output_frames, target_frames, batch_images, 
                                                              starting_point, num_input_frames, future_frame_idx,
                                                              num_channels, device, training)
                # output & target_frames size is [batches, num_channels * (n + 1), 128, 128]

            if debug:
                print('batch_num %d\tfuture_frame_idx %d\ttype %s' % (batch_num, future_frame_idx, prop_type))
                # print(output_frames.size(), target_frames.size())

            for ba in range(output_frames.size()[0]):
                score_keeper.add(output_frames[ba, -num_channels:, :, :].cpu(), 
                                 target_frames[ba, -num_channels:, :, :].cpu(), 
                                 future_frame_idx,"pHash", "pHash2", "SSIM", "Own", "RMSE")

            # if  batch_num == 1 and (((future_frame_idx + 1) % plot_frequency) == 0 or (future_frame_idx == 0)):

        logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(test_dataloader)))
        if debug: break
    plot_predictions(show_plots)
    plot_cutthrough(direction="Horizontal", location=None, show_plots=show_plots)