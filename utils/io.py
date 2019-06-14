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
    
def make_folder_results(folder_name):
    os.mkdir(folder_name)
    os.mkdir(os.path.join(folder_name,'figures'))

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