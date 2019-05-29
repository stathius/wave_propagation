import pickle
import os
"""
Saving and loading of figures, network state and other .pickle objects
"""
def save_network(obj, filename, device):
    network_dict = obj.cpu().state_dict()
    save(obj=network_dict, filename=filename)
    obj.to(device)

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
    if smoothen:
        inp = ndimage.gaussian_filter(inp, sigma=(.5, .5, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if obj is not None:
        obj.imshow(inp)
        obj.axis("off")
        if title is not None:
            obj.set_title(title)
    else:
        plt.imshow(inp)
        plt.axis("off")
        if title is not None:
            plt.title(title)
    if return_np:
        return inp