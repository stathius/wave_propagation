import pickle
import os
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
from utils.format import normalize_image

"""
Saving and loading of figures, network state and other .pickle objects
"""

def load_experiment():
    filename_data = os.path.join(results_dir,"all_data.pickle")
    logging.info('Loading datasets')
    all_data = load(filename_data)
    train_dataset = all_data["Training data"]
    val_dataset = all_data["Validation data"]
    test_dataset = all_data["Testing data"]

def save_network(model, filename):
    if hasattr(model, 'module'):
        network_dict = model.module.state_dict()
    else:
        network_dict = model.state_dict()
    torch.save(network_dict, filename)


def load_network(model, filename):
    dct = torch.load(filename, map_location='cpu')
    try:
        model.load_state_dict(dct)
    except:
        raise Warning('model and dictionary mismatch')
    return model

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

def create_results_folder(base_folder, experiment_name):
    exp_folder=os.path.join(base_folder, "experiments_results/")
    if not os.path.isdir(exp_folder):
        os.mkdir(exp_folder)

    results_dir = os.path.join(exp_folder, experiment_name)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    csv_dir = os.path.join(results_dir, 'csv/')
    if not os.path.isdir(csv_dir):
        os.mkdir(csv_dir)

    pickle_dir = os.path.join(results_dir, 'pickles/')
    if not os.path.isdir(pickle_dir):
        os.mkdir(pickle_dir)

    models_dir = os.path.join(results_dir, 'saved_models/')
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    predictions_dir = os.path.join(results_dir, 'predicted_frames/')
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    charts_dir = os.path.join(results_dir, 'charts_dir/')
    if not os.path.isdir(charts_dir):
        os.mkdir(charts_dir)

    return results_dir, csv_dir, pickle_dir, models_dir, predictions_dir, charts_dir

def imshow(image, title=None, smoothen=False, return_np=False, obj=None, normalize=None):
    """Imshow for Tensor."""
    num_channels = image.size()[0]

    if num_channels == 3:
        image = image.numpy().transpose((1, 2, 0))
        smooth_filter = (.5, .5, 0)
    elif num_channels == 1:
        image = image[0,:,:].numpy()
        smooth_filter = (.5, .5)
    else:
        raise Exception('Image size not supported ', image.size())

    if smoothen:
        image = ndimage.gaussian_filter(image, sigma=smooth_filter)

    if normalize is not None:
        image = normalize_image(image, normalize)

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