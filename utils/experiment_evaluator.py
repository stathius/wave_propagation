import time
import matplotlib.pyplot as plt
import math
import torch
import logging
import seaborn as sns
import pandas as pd
import numpy as np
from skimage import measure  # supports video also
from scipy.spatial import distance
from PIL import Image
import imagehash
import os
from utils.helper_functions import hex_str2bool, normalize_image
from utils.plotting import save_prediction_plot, save_cutthrough_plot
from utils.io import save, save_json, save_figure


def save_sequence_plots(sequence_index, starting_point, output_frames, target_frames, figures_dir, normalize, prediction=True, cutthrough=True):
    num_total_frames = output_frames.size(1)
    for frame_index in range(0, num_total_frames, 10):
        title = 'seq_%02d_start_%02d_frame_%02d' % (sequence_index, starting_point, frame_index)
        output = output_frames[0, frame_index, :, :].cpu().numpy()
        target = target_frames[0, frame_index, :, :].cpu().numpy()
        if prediction:
            save_prediction_plot(title, output, target, normalize, figures_dir)
        if cutthrough:
            save_cutthrough_plot(title, output, target, normalize, figures_dir, direction='Horizontal', location=None)


def get_test_predictions_pairs(model, batch_images, starting_point, num_total_output_frames):
    model.eval()
    num_input_frames = model.get_num_input_frames()
    with torch.no_grad():
        input_end_point = starting_point + num_input_frames
        input_frames = batch_images[:1, starting_point:input_end_point, :, :].clone()
        output_frames = model.get_future_frames(input_frames, num_total_output_frames)
        target_frames = batch_images[:1, input_end_point:(input_end_point + num_total_output_frames), :, :]
    return output_frames, target_frames


def get_sample_predictions(model, dataloader, device, figures_dir, normalizer, debug):
    time_start = time.time()
    num_input_frames = model.get_num_input_frames()
    num_output_frames = model.get_num_output_frames()
    for batch_num, batch_images in enumerate(dataloader):
        num_total_frames = batch_images.size(1)
        batch_images = batch_images.to(device)

        for starting_point in range(0, num_total_frames - 10 - num_input_frames, 10):
            num_total_output_frames = math.floor(math.floor((num_total_frames - num_input_frames - starting_point) / num_output_frames) * num_output_frames / 10) * 10  # requests multiple of ten

            output_frames, target_frames = get_test_predictions_pairs(model, batch_images, starting_point, num_total_output_frames)

            save_sequence_plots(batch_num, starting_point, output_frames, target_frames, figures_dir, normalizer)

            if debug:
                break
        if batch_num > 2:  # just plot couple of batches
            break

    logging.info('Sample predictions finished in %.1fs' % (time.time() - time_start))


def get_train_val_plots():
    pass


class Evaluator():
    """
    Calculates and keeps track of testing results
    SSIM/pHash/RMSE etc.
    """
    def __init__(self, starting_point, normalizer):
        super(Evaluator, self).__init__()
        self.starting_point = starting_point
        self.normalizer = normalizer

        self.intermitted = []
        self.frame = []
        self.hue = []

        self.state = {"pHash_val": [],
                      "pHash_frame": [],
                      "pHash_hue": [],
                      "pHash2_val": [],
                      "pHash2_frame": [],
                      "pHash2_hue": [],
                      "SSIM_val": [],
                      "SSIM_frame": [],
                      "SSIM_hue": [],
                      "SSIM_baseline_val": [],
                      "SSIM_baseline_frame": [],
                      "SSIM_baseline_hue": [],
                      "MSE_val": [],
                      "MSE_frame": [],
                      "MSE_hue": [],
                      "MSE_baseline_val": [],
                      "MSE_baseline_frame": [],
                      "MSE_baseline_hue": []
                      }

        self.own = False
        self.phash = False
        self.SSIM = False
        self.MSE = False
        self.phash2 = False

    def save_to_file(self, file):
        save(self, file)
        # save_json(self, file + '.json')
        save_json(self.state, file + '.state.json')

    def compute_experiment_metrics(self, exp, num_total_output_frames, debug=False):
        exp.model.eval()
        input_end_point = self.starting_point + exp.model.get_num_input_frames()
        with torch.no_grad():
            for batch_num, batch_images in enumerate(exp.dataloaders['test']):
                logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(exp.dataloaders['test'])))
                batch_images = batch_images.to(exp.device)

                input_frames = batch_images[:, self.starting_point:input_end_point, :, :]
                output_frames = exp.model.get_future_frames(input_frames, num_total_output_frames)
                num_real_output_frames = output_frames.size(1)
                target_frames = batch_images[:, input_end_point:(input_end_point + num_real_output_frames), :, :]

                self.compare_output_target(output_frames, target_frames)

                if debug:
                    print('batch_num %d\tSSIM %f' % (batch_num, self.state['SSIM_val'][-1]))
                    break

    def compare_output_target(self, output_frames, target_frames):
        batch_size = output_frames.size(0)
        num_output_frames = output_frames.size(1)
        for batch_index in range(batch_size):
            for frame_index in range(num_output_frames):
                output = output_frames[batch_index, frame_index, :, :].cpu().numpy()
                target = target_frames[batch_index, frame_index, :, :].cpu().numpy()
                self.add(output, target, frame_index, "pHash", "pHash2", "SSIM", "Own", "RMSE")
                if frame_index == 0:
                    dummy_prediction = target  # previous frame predicts the next
                elif frame_index > 0:
                    self.add_baseline(dummy_prediction, target, frame_index)
                    dummy_prediction = target

    def add_baseline(self, predicted, target, frame_nr):
        self.state['SSIM_baseline_val'].append(self.ssim(predicted, target))
        self.state['SSIM_baseline_frame'].append(frame_nr)
        self.state['SSIM_baseline_hue'].append("SSIM baseline")
        self.state['MSE_baseline_val'].append(np.sqrt(measure.compare_mse(predicted, target)))
        self.state['MSE_baseline_frame'].append(frame_nr)
        self.state['MSE_baseline_hue'].append("RMSE baseline")

    def add(self, predicted, target, frame_nr, *args):
        # input H * W
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
            self.state['SSIM_val'].append(ssim_score)
            self.state['SSIM_frame'].append(frame_nr)
            self.state['SSIM_hue'].append("SSIM")
            self.SSIM = True

        if "RMSE" in args:
            self.state['MSE_val'].append(np.sqrt(measure.compare_mse(predicted, target)))
            self.state['MSE_frame'].append(frame_nr)
            self.state['MSE_hue'].append("RMSE")
            self.MSE = True

        if "pHash" in args:
            self.state['pHash_val'].append(self.pHash(predicted, target, "hamming"))
            self.state['pHash_frame'].append(frame_nr)
            self.state['pHash_hue'].append("pHash - hamming")
            self.phash = True

        if "pHash2" in args:
            self.state['pHash2_val'].append(self.pHash(predicted, target, "jaccard"))
            self.state['pHash2_frame'].append(frame_nr)
            self.state['pHash2_hue'].append("pHash - jaccard")
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

    def ssim(self, predicted, target):
        return measure.compare_ssim(predicted, target, multichannel=False, gaussian_weights=True)

    def prepro(self, image):
        image = normalize_image(image, self.normalizer)
        # image = np.clip(image, 0, 1)
        return image

    def score(self, predicted, target):
        predicted_mean = np.mean(predicted, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        pred_relative = np.abs(predicted - predicted_mean)
        target_relative = np.abs(target - target_mean)

        relative_diff = np.mean(np.abs(pred_relative - target_relative)) / (np.sum(target_relative) / np.prod(np.shape(target)))

        absolute_diff = np.mean(np.abs(predicted - target)) / (np.sum(target) / np.prod(np.shape(target)))
        return relative_diff, absolute_diff

    def save_metrics_plots(self, output_dir):
        if self.own:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.frame, "Difference": self.intermitted, "Scoring Type": self.hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Difference", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "Scoring_Quality_start_%02d" % self.starting_point), obj=fig)

        if self.SSIM:
            all_data = {}
            all_data.update(
                {"Time-steps Ahead": self.state['SSIM_frame'] + self.state['SSIM_baseline_frame'],
                 "Similarity": self.state['SSIM_val'] + self.state['SSIM_baseline_val'],
                 "Scoring Type": self.state['SSIM_hue'] + self.state['SSIM_baseline_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Similarity", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "SSIM_Quality_start_%02d" % self.starting_point), obj=fig)

        if self.MSE:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.state['MSE_frame'] + self.state['MSE_baseline_frame'],
                             "Root Mean Square Error (L2 residual)": self.state['MSE_val'] + self.state['MSE_baseline_val'],
                             "Scoring Type": self.state['MSE_hue'] + self.state['MSE_baseline_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Root Mean Square Error (L2 residual)", hue="Scoring Type", data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "RMSE_Quality_start_%02d" % self.starting_point), obj=fig)

        if self.phash:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.state['pHash_frame'], "Hamming Distance": self.state['pHash_val'], "Scoring Type": self.state['pHash_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Hamming Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "Scoring_Spatial_Hamming_start_%02d" % self.starting_point), obj=fig)

        if self.phash2:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.state['pHash2_frame'], "Jaccard Distance": self.state['pHash2_val'], "Scoring Type": self.state['pHash2_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Jaccard Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "Scoring_Spatial_Jaccard_start_%02d" % self.starting_point), obj=fig)
        plt.close()
