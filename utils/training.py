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
from utils.io import imshow

debug=True

### COMMON FUNCTIONS TRAIN/VAL/TEST
def initial_input(model, batch_images, starting_point, num_input_frames, channels, device, training):
    """
    var           size
    batch_images  [16, 100, 128, 128]
    input_frames  [16, 5, 128, 128]
    output_frames  [16, 1, 128, 128]
    target_frames  [16, 1, 128, 128]
    """
    input_frames = batch_images[:, starting_point * channels:(starting_point + num_input_frames) * channels, :, :].to(device)
    output_frames = model(input_frames, mode='initial_input', training=training)
    target_index = starting_point + num_input_frames
    target_frames = batch_images[:, target_index * channels:(target_index + 1) * channels, :, :].to(device)
    return output_frames, target_frames

def reinsert(model, batch_images, starting_point, num_input_frames, num_output_frames, reinsert_offset, output_frames, target_frames, channels, device, training):
    output_frames = torch.cat((output_frames, model(output_frames[:, -num_input_frames * channels:, :, :].clone(), mode="reinsert", training=training)), dim=1)
    target_index = starting_point + reinsert_offset + num_input_frames
    target_frames = torch.cat((target_frames, batch_images[:, target_index * channels:(target_index + 1) * channels, :, :].to(device)), dim=1)
    return output_frames, target_frames

def propagate(model, batch_images, starting_point, num_input_frames, current_frame_index, output_frames, target_frames, channels, device, training):
    output_frames = torch.cat((output_frames, model(torch.Tensor([0]), mode="propagate", training=training)), dim=1)
    target_index = starting_point + current_frame_index + num_input_frames
    target_frames = torch.cat((target_frames, batch_images[:, target_index * channels:(target_index + 1) * channels, :, :].to(device)), dim=1)
    return output_frames, target_frames

def plot_predictions(output_frames, target_frames, channels):
    predicted = output_frames[i, -channels:, :, :].cpu().detach()
    des_target_frames = target_frames[i, -channels:, :, :].cpu().detach()
    fig = plt.figure(figsize=[8,8])
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted smoothened %02d" % n, smoothen=True, obj=pred)
    tar = fig.add_subplot(1, 2, 2)
    imshow(des_target_frames, title="Target %02d" % n, obj=tar)
    plt.show()


def train_epoch(model, lr_scheduler, epoch, train_dataloader, val_dataloader, num_input_frames, num_output_frames, reinsert_offset, channels, device, analyser, plot=False,):
    """
    Training of the network
    :param train: Training data
    :param val_dataloader: Validation data
    :return:
    """
    training = True
    model.train()           # initialises training stage/functions
    mean_loss = 0
    logging.info('Training: Ready to load batches')
    for batch_num, batch in enumerate(train_dataloader):
        batch_start = time.time()
        # logging.info('Batch: %d loaded in %.3f' %(batch_num, batch_time))
        mean_batch_loss = 0
        random_starting_points = random.sample(range(100 - num_input_frames - num_output_frames - 1), 10)
        batch_images = batch["image"]
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=batch_images.size()[0], training=True)
            lr_scheduler.optimizer.zero_grad()
            for current_frame_index in range(num_output_frames):
                if current_frame_index == 0:
                    output_frames, target_frames = initial_input(model, batch_images, starting_point, num_input_frames, channels, device, training=training)
                elif current_frame_index == reinsert_offset:
                    output_frames, target_frames = reinsert(model, batch_images, starting_point, num_input_frames, num_output_frames, reinsert_offset, output_frames, target_frames, channels, device, training=training)
                else:
                    output_frames, target_frames = propagate(model, batch_images, starting_point, num_input_frames, current_frame_index, output_frames, target_frames, channels, device, training=training)
                if plot and (i == 0) and (batch_num == 0):
                    plot_predictions(output_frames, target_frames, channels)
            loss = F.mse_loss(output_frames, target_frames)
            loss.backward()
            lr_scheduler.optimizer.step()

            mean_batch_loss += loss.item()
            if debug: break

        analyser.save_loss_batchwise(mean_batch_loss / (i + 1), batch_increment=1)
        mean_loss += loss.item()

        batch_time = time.time() - batch_start
        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime {:.2f}".format(epoch, batch_num + 1,
                   len(train_dataloader), 100. * (batch_num + 1) / len(train_dataloader), loss.item(), batch_time ) )        

        if debug and batch_num > 2:
            print('break')
            break

    epoch_loss = mean_loss / (batch_num + 1)
    return epoch_loss


def validate(model, val_dataloader, num_input_frames, num_output_frames ,reinsert_offset, channels, device, plot=False):
    """
    Validation of network (same protocol as training)
    :param val_dataloader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    val_start = time.time()
    training = False
    model.eval()
    overall_loss = 0
    for batch_num, batch in enumerate(val_dataloader):
        random_starting_points = random.sample(range(100 - num_input_frames - num_output_frames - 1), 10)
        batch_images = batch["image"]
        batch_loss = 0
        for i, starting_point in enumerate(random_starting_points):
            model.reset_hidden(batch_size=batch_images.size()[0], training=False)
            for current_frame_index in range(num_output_frames):
                if current_frame_index == 0:
                    output_frames, target_frames = initial_input(model, batch_images, starting_point, num_input_frames, channels, device, training=training)
                elif current_frame_index == reinsert_offset:
                    output_frames, target_frames = reinsert(model, batch_images, starting_point, num_input_frames, num_output_frames, reinsert_offset, output_frames, target_frames, channels, device, training=training)
                else:
                    output_frames, target_frames = propagate(model, batch_images, starting_point, num_input_frames, current_frame_index, output_frames, target_frames, channels, device, training=training)
                if plot and (i == 0) and (batch_num == 0):
                    plot_predictions(output_frames, target_frames, channels)
            batch_loss += F.mse_loss(output_frames, target_frames).item()
        overall_loss += batch_loss / (i + 1)
        if debug: break
    val_loss = overall_loss / (batch_num + 1)
    val_time = time.time() - val_start
    logging.info('Validation loss: %.6f\tTime: %.3f' % (val_loss, val_time))
    return val_loss


def test(model, test_dataloader, num_input_frames, num_output_frames, channels, device, score_keeper, results_dir, plot=True):
    """
    Testing of network
    :param test_dataloader: Data to test
    :param plot: If to plot predictionss
    :return:
    """
    def initial_input(no_more_target):
        output = model(input_frames.to(device))
        try:
            target = batch_images[:, (starting_point + some_counter + num_input_frames) * channels:(starting_point + some_counter + num_input_frames + 1) * channels, :, :].to(device)
        except Exception as e:
            print(e)
            no_more_target = True
            target = None
        return output, target, no_more_target

    def reinsert(output, target, no_more_target):
        output = torch.cat((output, model(input_frames, mode="reinsert")), dim=1)
        try:
            target = torch.cat((target, 
                batch_images[:, (starting_point + some_counter + num_input_frames) * channels:(starting_point + some_counter + num_input_frames + 1) * channels, :, :].to(device)), dim=1)
        except Exception as e:
            print(e)
            no_more_target = True
        return output, target, no_more_target

    def propagate(output, target, no_more_target):
        if current_frame_index < (num_output_frames - refeed_offset):
            output = torch.cat((output, model(torch.Tensor([0]), mode="propagate")), dim=1)
            try:
                target = torch.cat((target, 
                    batch_images[:, (starting_point + some_counter + num_input_frames) * channels:(starting_point + some_counter + num_input_frames + 1) * channels, :, :].to(device)), dim=1)
            except:
                no_more_target = True
        return output, target, no_more_target

    def plot_predictions():
        if (total == 0) & (current_frame_index == 0) & (run == 0):
            for imag in range(int(input_frames.shape[1] / channels)):
                fig = plt.figure().add_axes()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                sns.set_context("talk")
                imshow(input_frames[selected_batch, imag * channels:(imag + 1) * channels, :, :], title="Input %01d" % imag, obj=fig)
                figure_save(results_dir + "Input %02d" % imag)
        if (total == 0) & (current_frame_index < (num_output_frames - refeed_offset)):
            predicted = output[selected_batch, -channels:, :, :].cpu()
            des_target = target[selected_batch, -channels:, :, :].cpu()
            fig = plt.figure()
            sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
            sns.set_context("talk")
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted %02d" % some_counter, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % target_counter, obj=tar)
            figure_save(results_dir + "Prediction %02d" % some_counter)
            plt.show()

    def plot_cutthrough(frequently_plot=5, direction="Horizontal", location=None):
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
            #g = sns.FacetGrid(pd.DataFrame.from_dict(data_dict), col="Image")
            #g.map(sns.lineplot, "Pixel Location", "Intensity")
            sns.lineplot(x="Pixel Location", y="Intensity", hue="Image",
                         data=pd.DataFrame.from_dict(data_dict), ax=profile)
            profile.set_title("Intensity Profile")

        if total == 0:
            if ((some_counter + 1) % frequently_plot) == 0 or (some_counter == 0):
                predicted = output[selected_batch, -channels:, :, :].cpu()
                des_target = target[selected_batch, -channels:, :, :].cpu()
                fig = plt.figure()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                with sns.axes_style("white"):
                    pre = fig.add_subplot(2, 2, 1)
                    tar = fig.add_subplot(2, 2, 2)
                with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
                    profile = fig.add_subplot(2, 2, (3, 4))

                predicted = imshow(predicted, title="Predicted %02d" % some_counter, return_np=True, obj=pre)
                des_target = imshow(des_target, title="Target %02d" % target_counter, return_np=True, obj=tar)
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

                cutthrough(predicted, des_target, "Predicted", "Target")
                figure_save(results_dir + "Cut-through %02d" % some_counter, obj=fig)
                plt.show()


    model.eval()
    starting_point = 15 # Can be 0
    refeed_offset = 0
    selected_batch = random.randint(0, 15)
    if (num_output_frames - refeed_offset) < num_input_frames:
        refeed_offset = num_output_frames - num_input_frames
    for batch_num, batch in enumerate(test_dataloader):
        batch_images = batch["image"]
        input_frames = batch_images[:, starting_point * channels:(starting_point + num_input_frames) * channels, :, :]
        model.reset_hidden(input_frames.size()[0])
        no_more_target = False
        some_counter = target_counter = 0
        for run in range(int(math.ceil((100 - (starting_point + num_input_frames + 1)) / (num_output_frames - refeed_offset)))):
            if run != 0:
                if (refeed_offset == 0) or ((num_output_frames - refeed_offset) <= num_input_frames):
                    input_frames = output[:, -num_input_frames * channels:, :, :]
                else:
                    input_frames = output[:, -(num_input_frames + refeed_offset) * channels:-refeed_offset * channels, :, :]
                some_counter -= refeed_offset
            for current_frame_index in range(num_output_frames):
                if current_frame_index == 0:
                    if run == 0:
                        output, target, no_more_target = initial_input(no_more_target)
                    else:
                        output, target, no_more_target = reinsert(output, target, no_more_target)
                else:
                    output, target, no_more_target = propagate(output, target, no_more_target)
                    # output & target size is [batches, channels * (n + 1), 128, 128]

                if (not no_more_target) & (current_frame_index < (num_output_frames - refeed_offset)):
                    for ba in range(output.size()[0]):
                        score_keeper.add(output[ba, -channels:, :, :].cpu(), target[ba, -channels:, :, :].cpu(), 
                                         some_counter,"pHash", "pHash2", "SSIM", "Own", "RMSE")

                if plot:
                    plot_predictions()
                    plot_cutthrough()
                some_counter += 1
                if not no_more_target:
                    target_counter = copy.copy(some_counter)

        total += target.size()[0]
        logging.info("{:d} out of {:d}".format(batch_num + 1, len(test_dataloader)))
        if debug: break