import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import seaborn as sns
import pandas as pd
import numpy as np
import os
from utils.io import figure_save


def imshow(image, title=None, smoothen=False, return_np=False, obj=None, normalize=None):
    """Imshow for Tensor."""
    num_channels = image.size()[0]

    if num_channels == 3:
        image = image.numpy().transpose((1, 2, 0))
        smooth_filter = (.5, .5, 0)
    elif num_channels == 1:
        image = image[0, :, :].numpy()
        smooth_filter = (.5, .5)
    else:
        raise Exception('Image size not supported ', image.size())

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
        figure_save(os.path.join(figures_dir, "Input %02d" % imag))


def plot_predictions(future_frame_idx, input_frames, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots):
    # -1 means print last frame
    predicted = output_frames[image_to_plot, -1:, :, :].cpu()
    target = target_frames[image_to_plot, -1:, :, :].cpu()
    fig = plt.figure()
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("talk")
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted %02d" % future_frame_idx, smoothen=True, obj=pred, normalize=normalize)
    tar = fig.add_subplot(1, 2, 2)
    imshow(target, title="Target %02d" % future_frame_idx, obj=tar, normalize=normalize)
    figure_save(os.path.join(figures_dir, "Prediction %02d" % future_frame_idx))
    if show_plots:
        plt.show()


def plot_cutthrough(future_frame_idx, output_frames, target_frames, image_to_plot, normalize, figures_dir, show_plots, direction, location=None):
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

    predicted = output_frames[image_to_plot, -1:, :, :].cpu()
    target = target_frames[image_to_plot, -1:, :, :].cpu()
    fig = plt.figure()
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    with sns.axes_style("white"):
        pre = fig.add_subplot(2, 2, 1)
        tar = fig.add_subplot(2, 2, 2)
    with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
        profile = fig.add_subplot(2, 2, (3, 4))

    predicted = imshow(predicted, title="Predicted %02d" % future_frame_idx, return_np=True, obj=pre, normalize=normalize)
    target = imshow(target, title="Target %02d" % future_frame_idx, return_np=True, obj=tar, normalize=normalize)
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
    figure_save(os.path.join(figures_dir, "Cut-through %02d" % future_frame_idx), obj=fig)
    if show_plots:
        plt.show()