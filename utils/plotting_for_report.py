import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from . plotting import imshow

def get_cutthrough_plot_report(exp_name, frame_num, predicted, target, direction, location=None, one_line=False):
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
        # profile.set_title("Intensity Profile")
        plt.tight_layout()


    fig = plt.figure()
    sns.set_context("paper", font_scale=2)

    if one_line:
        with sns.axes_style("white"):
            pre = fig.add_subplot(1, 4, 1)
            tar = fig.add_subplot(1, 4, 2)
        with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
            profile = fig.add_subplot(1, 4, (3, 4))
        exp_name += '_one_line'
        fsize=(12,3)
    else:
        with sns.axes_style("white"):
            pre = fig.add_subplot(2, 2, 1)
            tar = fig.add_subplot(2, 2, 2)
        with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
            profile = fig.add_subplot(2, 2, (3, 4))
        fsize=(6,6)

    scale = 0.45
    sns.set(rc={'figure.figsize':fsize,
                "lines.linewidth": 1.5,
                'figure.titlesize': 40*scale,
            "axes.labelsize": 40*scale,
            'ytick.labelsize': 100*scale,
            'xtick.labelsize': 100*scale,
            'legend.fontsize': 25*scale,
            'legend.borderpad': 0.2,
            'legend.columnspacing': 0.5,
            'legend.labelspacing':0.3,
            'legend.borderaxespad': 0.3})




    predicted = imshow(predicted, title="Predicted %d" %frame_num, return_np=True, obj=pre)
    target = imshow(target, title="Target %d" %frame_num, return_np=True, obj=tar)
    # pre.imshow(predicted, cmap='gray', interpolation='none')
    # plt.xlabel("Predicted %d" %frame_num)
    # tar.imshow(target, cmap='gray', interpolation='none')
    # plt.xlabel("Target %d" %frame_num)
    # pre.axis("off")
    # tar.axis("off")


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

    plt.tight_layout()
    plt.savefig('qualitative/cutthrough/%s_%s.pdf'% (exp_name, frame_num), format='pdf')
    plt.close()
