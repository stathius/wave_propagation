import pickle
import os
import torch 
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np

"""
Saving and loading of figures, network state and other .pickle objects
"""

def save_network(model, filename):
    if hasattr(model, 'module'):
        network_dict = model.module.state_dict()
    else:
        network_dict = model.state_dict()
    torch.save(network_dict, filename)


def load_network(model, device, filename):
    dct = torch.load(filename, map_location='cpu')
    try:
        model.load_state_dict(dct)
    except:
        raise Warning('model and dictionary mismatch')
    model.to(device)
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
    if os.path.isdir(folder_name):
        imgs = os.listdir(folder_name)
        for img in imgs:
            os.remove(folder_name + "/" + img)
    else:
        os.mkdir(folder_name)

def imshow(inp, title=None, smoothen=False, return_np=False, obj=None):
    """Imshow for Tensor."""
    channels = inp.size()[0]

    if channels == 3:
        inp = inp.numpy().transpose((1, 2, 0))
        smooth_filter = (.5, .5, 0)
    elif channels == 1:
        inp = inp[0,:,:].numpy()
        smooth_filter = (.5, .5)
    else:
        raise Exception('Image size not supported ', inp.size())

    if smoothen:
        inp = ndimage.gaussian_filter(inp, sigma=smooth_filter)

    inp = np.clip(inp, 0, 1)
    if obj is not None:
        obj.imshow(inp, cmap='gray', interpolation='none')
        obj.axis("off")
        if title is not None:
            obj.set_title(title)
    else:
        plt.imshow(inp, cmap='gray', interpolation='none')
        plt.axis("off")
        if title is not None:
            plt.title(title)
    if return_np:
        return inp