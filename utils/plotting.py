import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import seaborn as sns
import pandas as pd
import numpy as np
import os
from utils.io import save_figure


def imshow(image, title=None, smoothen=False, return_np=False, obj=None):
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


def save_prediction_plot(title, predicted, target, normalize, figures_dir):
    fig = plt.figure(figsize=(6, 6))
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("talk")
    pred = fig.add_subplot(1, 2, 1)
    imshow(predicted, title="Predicted", smoothen=True, obj=pred)
    tar = fig.add_subplot(1, 2, 2)
    imshow(target, title="Target", obj=tar)
    fig.suptitle(title)
    save_figure(os.path.join(figures_dir, title), fig)
    plt.close()


def get_cutthrough_plot(title, predicted, target, direction, location=None):
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

    fig = plt.figure(figsize=(12, 4))
    sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
    with sns.axes_style("white"):
        pre = fig.add_subplot(1, 4, 1)
        tar = fig.add_subplot(1, 4, 2)
    with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
        profile = fig.add_subplot(1, 4, (3, 4))

    predicted = imshow(predicted, title="Predicted", return_np=True, obj=pre)
    target = imshow(target, title="Target", return_np=True, obj=tar)
    if not location:
        if "Horizontal" in direction:
            std = np.std(target, axis=1)
        elif "Vertical" in direction:
            std = np.std(target, axis=0)
        stdmax = np.where(std.max() == std)[0]  # just keep one line
    else:
        stdmax = location

    if "Horizontal" in direction:
        pre.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
        tar.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
    elif "Vertical" in direction:
        pre.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")
        tar.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")

    cutthrough(predicted, target, "Predicted", "Target")
    fig.suptitle(title)
    return fig
