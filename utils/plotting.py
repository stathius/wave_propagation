import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import seaborn as sns
import pandas as pd
import numpy as np
import os
from utils.io import figure_save


def imshow(image, title=None, smoothen=False, return_np=False, obj=None, normalize=None):
    """Imshow for Tensor."""
    smooth_filter = (.5, .5)

    if smoothen:
        image = ndimage.gaussian_filter(image, sigma=smooth_filter)

    # image = np.clip(image, 0, 1)
    if obj is not None:
        obj.imshow(image, cmap='gray', interpolation='none')
        obj.axis("off")
        if title is not None:
            obj.set_title(title)
    else:
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.axis("off")
        if title is not None:
            plt.title(title)

    if return_np:
        return image


def plot_input_frames(input_frames, image_to_plot, normalize, figures_dir):
    # Plot the N first input frames
    for imag in range(int(input_frames.shape[1])):
        fig = plt.figure().add_axes()
        sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        imshow(input_frames[image_to_plot, imag:(imag + 1), :, :], title="Input %01d" % imag, obj=fig, normalize=normalize)
        figure_save(os.path.join(figures_dir, "Input_%02d" % imag))


def save_prediction_plot(batch_index, frame_index, predicted, target, normalize, figures_dir):
    # -1 means print last frame
    # predicted = predicted[image_to_plot, -1:, :, :].cpu()
    # target = target[image_to_plot, -1:, :, :].cpu()
    fig = plt.figure(figsize=(6, 6))
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("talk")
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted %02d" % frame_index, smoothen=True, obj=pred, normalize=normalize)
    tar = fig.add_subplot(1, 2, 2)
    imshow(target, title="Target %02d" % frame_index, obj=tar, normalize=normalize)
    figure_save(os.path.join(figures_dir, "Prediction_%03d_%03d" % (batch_index, frame_index)), fig)
    plt.close()

def save_cutthrough_plot(batch_index, frame_index, predicted, target, normalize, figures_dir, direction, location=None):
    def cutthrough(img1, img2, hue1, hue2):
        intensity = []
        location = []
        hue = []
        if "Horizontal" in direction:
            intensity = np.append(img1[stdmax[0], :], img2[stdmax[0], :])
            length1 = img1.shape[1]
            length2 = img2.shape[1]
            location = list(range(length1)) + list(range(length2))
            hue = [hue1] * length1 + [hue2] * length2
        elif "Vertical" in direction:
            intensity = np.append(img1[:, stdmax[0]], img2[:, stdmax[0]])
            width1 = img1.shape[0]
            width2 = img2.shape[0]
            location = list(range(width1)) + list(range(width2))
            hue = [hue1] * width1 + [hue2] * width2

        data_dict = {"Intensity": intensity, "Pixel Location": location, "Image": hue}
        sns.lineplot(x="Pixel Location", y="Intensity", hue="Image",
                     data=pd.DataFrame.from_dict(data_dict), ax=profile)
        profile.set_title("Intensity Profile")

    # predicted = predicted[image_to_plot, -1:, :, :].cpu()
    # target = target[image_to_plot, -1:, :, :].cpu()
    # fig = plt.figure()
    fig = plt.figure(figsize=(12, 4))
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    with sns.axes_style("white"):
        pre = fig.add_subplot(1, 4, 1)
        tar = fig.add_subplot(1, 4, 2)
    with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
        profile = fig.add_subplot(1, 4, (3, 4))

    predicted = imshow(predicted, title="Predicted %d" % frame_index, return_np=True, obj=pre, normalize=normalize)
    target = imshow(target, title="Target %d" % frame_index, return_np=True, obj=tar, normalize=normalize)
    if not location:
        if "Horizontal" in direction:
            std = np.std(target, axis=1)
        elif "Vertical" in direction:
            std = np.std(target, axis=0)
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
    cutthrough(predicted, target, "Predicted", "Target")
    figure_save(os.path.join(figures_dir, "Cut_through_batch_%03d_%03d" % (batch_index, frame_index)), obj=fig)
    plt.close()


def save_sequence_plots(batch_num, output_frames, target_frames, figures_dir, normalize):
    num_total_frames = output_frames.size(1)
    batch_index = 0  # doesn't really matter
    for frame_index in range(0, num_total_frames, 10):
        output = output_frames[batch_index, frame_index, :, :].cpu().numpy()
        target = target_frames[batch_index, frame_index, :, :].cpu().numpy()
        save_prediction_plot(batch_num, frame_index, output, target, normalize, figures_dir)
        save_cutthrough_plot(batch_num, frame_index, output, target, normalize, figures_dir, direction='Horizontal', location=None)
