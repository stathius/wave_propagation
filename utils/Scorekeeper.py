import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.io import figure_save
import numpy as np
from skimage import measure #supports video also
from scipy.spatial import distance
from PIL import Image
import imagehash
import os
from utils.format import hex_str2bool

class Scorekeeper():
    """
    Calculates and keeps track of testing results
    SSIM/pHash/RMSE etc.
    """
    def __init__(self, results_dir, channels):
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
        self.phash2 = False
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        self.channels = channels

    def add(self, predicted, target, frame_nr, *args):
        predicted = self.prepro(predicted)
        target = self.prepro(target)

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
            self.pHash_val.append(self.pHash(predicted, target, "hamming"))
            self.pHash_frame.append(frame_nr)
            self.pHash_hue.append("pHash - hamming")
            self.phash = True

        if "pHash2" in args:
            self.pHash2_val.append(self.pHash(predicted, target, "jaccard"))
            self.pHash2_frame.append(frame_nr)
            self.pHash2_hue.append("pHash - jaccard")
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
        # print(predicted.size)
        multichannel = self.channels > 1
        return measure.compare_ssim(predicted, target, multichannel=multichannel, gaussian_weights=True)

    def prepro(self, image):
        if image.size()[0]==3:
            image = image.numpy().transpose((1, 2, 0))
        else:
            image = image[0,:,:].numpy()
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image = std * image + mean
        # np.clip(image, 0, 1)
        return image

    def score(self, predicted, target):
        predicted_mean = np.mean(predicted, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        pred_relative = np.abs(predicted - predicted_mean)
        target_relative = np.abs(target - target_mean)

        relative_diff = np.mean(np.abs(pred_relative - target_relative))                         / (np.sum(target_relative) / np.prod(np.shape(target)))

        absolute_diff = np.mean(np.abs(predicted - target)) / (np.sum(target) / np.prod(np.shape(target)))

        return relative_diff, absolute_diff

    def plot(self):
        if self.own:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.frame, "Difference": self.intermitted, "Scoring Type": self.hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Difference", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig,  ci='sd')
            figure_save(os.path.join(self.figures_dir, "Scoring_Quality"), obj=fig)
            plt.show()

        if self.SSIM:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.SSIM_frame, "Similarity": self.SSIM_val,
                                  "Scoring Type": self.SSIM_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Similarity", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig,  ci='sd')
            # plt.ylim(0.0, 1)
            figure_save(os.path.join(self.figures_dir, "SSIM_Quality"), obj=fig)
            plt.show()

        if self.MSE:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.MSE_frame, "Root Mean Square Error (L2 residual)": self.MSE_val,
                                 "Scoring Type": self.MSE_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Root Mean Square Error (L2 residual)", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            figure_save(os.path.join(self.figures_dir, "RMSE_Quality"), obj=fig)
            plt.show()

        if self.phash:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.pHash_frame, "Hamming Distance": self.pHash_val,
                                   "Scoring Type": self.pHash_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Hamming Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            figure_save(os.path.join(self.figures_dir, "Scoring_Spatial_Hamming"), obj=fig)
            plt.show()

        if self.phash2:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.pHash2_frame, "Jaccard Distance": self.pHash2_val,
                                   "Scoring Type": self.pHash2_hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Jaccard Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            figure_save(os.path.join(self.figures_dir, "Scoring_Spatial_Jaccard"), obj=fig)
            plt.show()