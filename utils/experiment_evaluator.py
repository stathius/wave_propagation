import time
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import DataLoader
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
from utils.plotting import get_cutthrough_plot
from utils.io import save, save_json, save_figure
from utils.experiment import get_transforms, get_normalizer
from utils.WaveDataset import WaveDataset


def image_prepro(image, normalizer):
    image = normalize_image(image, normalizer)
    image = np.clip(image, 0, 1)
    return image


def get_test_predictions_pairs(model, belated, batch_images, starting_point, num_total_output_frames):
    model.eval()
    num_input_frames = model.get_num_input_frames()
    with torch.no_grad():
        input_end_point = starting_point + num_input_frames
        input_frames = batch_images[:1, starting_point:input_end_point, :, :].clone()
        output_frames = model.get_future_frames(input_frames, num_total_output_frames, belated)
        target_frames = batch_images[:1, input_end_point:(input_end_point + num_total_output_frames), :, :]
    return output_frames, target_frames


def save_sequence_plots(sequence_index, starting_point, output_frames, target_frames, figures_dir, normalizer, dataset_name):
    # Used also in experiment runner
    num_total_frames = output_frames.size(1)
    for frame_index in range(0, num_total_frames, 10):
        params = 'seq_%02d_start_%02d_frame_%02d' % (sequence_index, starting_point, frame_index + 1)
        output = image_prepro(output_frames[0, frame_index, :, :].cpu().numpy(), normalizer)
        target = image_prepro(target_frames[0, frame_index, :, :].cpu().numpy(), normalizer)
        title = "%s_Ver_%s" % (dataset_name, params)
        fig = get_cutthrough_plot(title, output, target, direction='Horizontal', location=None)
        save_figure(os.path.join(figures_dir, title), obj=fig)
        title = "%s_Hor_%s" % (dataset_name, params)
        fig = get_cutthrough_plot(title, output, target, direction='Vertical', location=None)
        plt.close()


def get_sample_predictions(model, belated, dataloader, dataset_name, device, figures_dir, normalizer, debug):
    time_start = time.time()
    num_input_frames = model.get_num_input_frames()
    for batch_num, batch_images in enumerate(dataloader):
        num_total_frames = batch_images.size(1)
        batch_images = batch_images.to(device)

        for starting_point in range(0, max(21, num_total_frames - num_input_frames), 10):
            num_total_output_frames = math.floor(math.floor((num_total_frames - num_input_frames - starting_point)) / 10) * 10  # requests multiple of ten
            if num_total_output_frames < 10:
                continue

            output_frames, target_frames = get_test_predictions_pairs(model, belated, batch_images, starting_point, num_total_output_frames)
            save_sequence_plots(batch_num, starting_point, output_frames, target_frames, figures_dir, normalizer, dataset_name)

            if debug:
                break
        if debug:
            break
        if batch_num > 2:  # just plot couple of batches
            break

    logging.info('Sample predictions finished in %.1fs' % (time.time() - time_start))


def create_evaluation_dataloader(data_directory, normalizer_type):
    logging.info('Creating evaluation dataset %s' % data_directory)
    transform = get_transforms(get_normalizer(normalizer_type))

    classes = [dI for dI in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, dI))]
    # classes = os.listdir(data_directory)
    imagesets = []
    for cla in classes:
        im_list = sorted(os.listdir(data_directory + cla))
        imagesets.append((im_list, cla))

    dataset_info = [data_directory, classes, imagesets]
    dataset = WaveDataset(dataset_info, transform["Test"])
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


def evaluate_experiment(experiment, args_new):
    start_time = time.time()
    logging.info("Start testing")
    dataloaders = {"Test": experiment.dataloaders['test'],
                   "Lines": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Lines/'), experiment.args.normalizer_type),
                   "Double_Drop": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Double_Drop/'), experiment.args.normalizer_type),
                   "Illumination_135": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Illumination_135/'), experiment.args.normalizer_type),
                   "Shallow_Depth": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Shallow_Depth/'), experiment.args.normalizer_type),
                   "Smaller_Tub": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Smaller_Tub/'), experiment.args.normalizer_type),
                   "Bigger_Tub": create_evaluation_dataloader(os.path.join(experiment.dirs['data_base'], 'Bigger_Tub/'), experiment.args.normalizer_type)
                   }

    for dataset_name, dataloader in dataloaders.items():
        logging.info("Evaluating dataset: %s" % dataset_name)
        evaluator = Evaluator(args_new.test_starting_point, dataset_name, experiment.normalizer)
        evaluator.compute_experiment_metrics(experiment.model, experiment.args_new.belated, dataloader, args_new.num_total_output_frames, experiment.device, debug=args_new.debug)
        evaluator.save_metrics_plots(experiment.dirs['charts'])
        evaluator.save_to_file(experiment.files['evaluator'] % (dataset_name, args_new.test_starting_point))
        # Get the sample plots after you compute everything else because the dataloader iterates from the beginning
        if args_new.get_sample_predictions:
            logging.info("Generate prediction plots for %s" % dataset_name)
            get_sample_predictions(experiment.model, args_new.belated, dataloader, dataset_name, experiment.device, experiment.dirs['predictions'], experiment.normalizer, args_new.debug)
        logging.info('Elapsed time: %.0f' % (time.time() - start_time))
        if args_new.debug:
            break


class Evaluator():
    """
    Calculates and keeps track of testing results
    SSIM/pHash/RMSE etc.
    """
    def __init__(self, starting_point, dataset_name, normalizer):
        super(Evaluator, self).__init__()
        self.starting_point = starting_point
        self.normalizer = normalizer
        self.dataset_name = dataset_name

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
                      "SSIM_previous_frame_val": [],
                      "SSIM_previous_frame_frame": [],
                      "SSIM_previous_frame_hue": [],
                      "SSIM_last_input_val": [],
                      "SSIM_last_input_frame": [],
                      "SSIM_last_input_hue": [],
                      "MSE_val": [],
                      "MSE_frame": [],
                      "MSE_hue": [],
                      "MSE_previous_frame_val": [],
                      "MSE_previous_frame_frame": [],
                      "MSE_previous_frame_hue": [],
                      "MSE_last_input_val": [],
                      "MSE_last_input_frame": [],
                      "MSE_last_input_hue": [],
                      }

        self.own = False
        self.phash = False
        self.SSIM = False
        self.MSE = False
        self.phash2 = False

    def get_rmse_values(self):
        df = pd.DataFrame.from_dict({'RMSE': self.state['MSE_val'],
                                       'Frames': self.state['MSE_frame']}).groupby('Frames', as_index=False).agg(['mean', 'std'])
        return df['RMSE']

    def get_ssim_values(self):
        df = pd.DataFrame.from_dict({'SSIM': self.state['SSIM_val'],
                                       'Frames': self.state['SSIM_frame']}).groupby('Frames', as_index=False).agg(['mean', 'std'])
        return df['SSIM']

    def save_to_file(self, file):
        save(self, file)
        # save_json(self, file + '.json')
        save_json(self.state, file + '.state.json')

    def compute_experiment_metrics(self, model, belated, dataloader, num_total_output_frames, device, debug=False):
        model.eval()
        input_end_point = self.starting_point + model.get_num_input_frames()
        with torch.no_grad():
            for batch_num, batch_images in enumerate(dataloader):
                logging.info("Testing batch {:d} out of {:d}".format(batch_num + 1, len(dataloader)))
                # batch_images = batch_images.to(device)

                target_frames = batch_images[:, input_end_point:(input_end_point + num_total_output_frames), :, :].cpu().numpy()
                last_input = batch_images[:, (input_end_point - 1):input_end_point, :, :].cpu().numpy()
                num_real_output_frames = target_frames.shape[1]

                # logging.info('num_real_output_frames %d' % num_real_output_frames)
                self.compare_output_target(
                    model.get_future_frames(batch_images[:, self.starting_point:input_end_point, :, :].to(device),
                                            num_real_output_frames, belated).cpu().numpy(), target_frames, last_input)

                if debug:
                    print('batch_num %d\tSSIM %f' % (batch_num, self.state['SSIM_val'][-1]))
                    break

    def compare_output_target(self, output_frames, target_frames, last_input_batch):
        batch_size = output_frames.shape[0]
        num_output_frames = output_frames.shape[1]
        for batch_index in range(batch_size):
            for frame_index in range(num_output_frames):
                outpu = image_prepro(output_frames[batch_index, frame_index, :, :], self.normalizer)
                target = image_prepro(target_frames[batch_index, frame_index, :, :], self.normalizer)
                self.add(outpu, target, frame_index, "pHash", "pHash2", "SSIM", "Own", "RMSE")
                if frame_index == 0:
                    previous_frame = target  # previous frame predicts the next
                elif frame_index > 0:
                    self.add_baseline('previous_frame', previous_frame, target, frame_index)
                    previous_frame = target
                last_input = image_prepro(last_input_batch[batch_index, 0, :, :], self.normalizer)
                self.add_baseline('last_input', last_input, target, frame_index)

    def add_baseline(self, name, predicted, target, frame_nr):
        self.state['SSIM_%s_val' % name].append(self.ssim(predicted, target))
        self.state['SSIM_%s_frame' % name].append(frame_nr)
        self.state['SSIM_%s_hue' % name].append("SSIM %s" % name.replace('_', ' ').title())
        self.state['MSE_%s_val' % name].append(np.sqrt(measure.compare_mse(predicted, target)))
        self.state['MSE_%s_frame' % name].append(frame_nr)
        self.state['MSE_%s_hue' % name].append("RMSE %s" % name.replace('_', ' ').title())

    def add(self, predicted, target, frame_nr, *args):
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

    def score(self, predicted, target):
        predicted_mean = np.mean(predicted, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        pred_relative = np.abs(predicted - predicted_mean)
        target_relative = np.abs(target - target_mean)

        relative_diff = np.mean(np.abs(pred_relative - target_relative)) / (np.sum(target_relative) / np.prod(np.shape(target)))

        absolute_diff = np.mean(np.abs(predicted - target)) / (np.sum(target) / np.prod(np.shape(target)))
        return relative_diff, absolute_diff

    def save_metrics_plots(self, output_dir):
        logging.info('saving metric plots to %s' % output_dir)
        if self.own:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.frame, "Difference": self.intermitted, "Scoring Type": self.hue})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Difference", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "%s_Scoring_Quality_start_%02d" % (self.dataset_name, self.starting_point)), obj=fig)

        if self.SSIM:
            all_data = {}
            all_data.update(
                {"Time-steps Ahead": self.state['SSIM_frame'] + self.state['SSIM_previous_frame_frame'] +  self.state['SSIM_last_input_frame'] ,
                 "Similarity": self.state['SSIM_val']         + self.state['SSIM_previous_frame_val']   +  self.state['SSIM_last_input_val']   ,
                 "Scoring Type": self.state['SSIM_hue']       + self.state['SSIM_previous_frame_hue']   +  self.state['SSIM_last_input_hue']   })
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Similarity", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "%s_SSIM_Quality_start_%02d" % (self.dataset_name, self.starting_point)), obj=fig)

        if self.MSE:
            all_data = {}
            all_data.update({"Time-steps Ahead":        self.state['MSE_frame'] + self.state['MSE_previous_frame_frame'] + self.state['MSE_last_input_frame'] ,
                             "Root Mean Square Error":  self.state['MSE_val']   + self.state['MSE_previous_frame_val']   + self.state['MSE_last_input_val']   ,
                             "Scoring Type":            self.state['MSE_hue']   + self.state['MSE_previous_frame_hue']   + self.state['MSE_last_input_hue']   })
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Root Mean Square Error", hue="Scoring Type", data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "%s_RMSE_Quality_start_%02d" % (self.dataset_name, self.starting_point)), obj=fig)

        if self.phash:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.state['pHash_frame'], "Hamming Distance": self.state['pHash_val'], "Scoring Type": self.state['pHash_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Hamming Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "%s_Scoring_Spatial_Hamming_start_%02d" % (self.dataset_name, self.starting_point)), obj=fig)

        if self.phash2:
            all_data = {}
            all_data.update({"Time-steps Ahead": self.state['pHash2_frame'], "Jaccard Distance": self.state['pHash2_val'], "Scoring Type": self.state['pHash2_hue']})
            fig = plt.figure().add_axes()
            sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
            sns.lineplot(x="Time-steps Ahead", y="Jaccard Distance", hue="Scoring Type",
                         data=pd.DataFrame.from_dict(all_data), ax=fig, ci='sd')
            save_figure(os.path.join(output_dir, "%s_Scoring_Spatial_Jaccard_start_%02d" % (self.dataset_name, self.starting_point)), obj=fig)
        plt.close()
