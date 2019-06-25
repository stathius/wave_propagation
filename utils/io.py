import pickle
import os
import csv
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from utils.format import normalize_image


def save_datasets_to_file(train_dataset, val_dataset, test_dataset, filename):
    all_data = {"Training data": train_dataset,
                "Validation data": val_dataset,
                "Testing data": test_dataset}
    save(all_data, filename)


def load_datasets_from_file(filename_data):
    all_data = load(filename_data)
    train_dataset = all_data["Training data"]
    val_dataset = all_data["Validation data"]
    test_dataset = all_data["Testing data"]
    return train_dataset, val_dataset, test_dataset

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
    dirs = {'base': base_folder}
    exp_folder=os.path.join(base_folder, "experiments_results/")
    if not os.path.isdir(exp_folder):
        os.mkdir(exp_folder)

    dirs['results'] = os.path.join(exp_folder, experiment_name)
    if not os.path.isdir(dirs['results']):
        os.mkdir(dirs['results'])

    for d in ['logs', 'pickles', 'models', 'predictions', 'charts']:
        dirs[d] = os.path.join(dirs['results'], '%s/' % d)
        if not os.path.isdir(dirs[d]):
            os.mkdir(dirs[d])
    return dirs

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

###### EXPERIMENT BUILDER STUFF FROM HERE

def save_stuff_json(experiment_log_dir, filename, stuff):
    """
    Saves stuff
    """
    import json
    filename = os.path.join(experiment_log_dir, filename)
    with open(filename, 'w') as f:
        f.write("%s" % json.dumps(stuff))
    f.close()

def read_stats(folder):
    df = pd.read_csv(os.path.join(folder,"summary.csv"))
    dicc = {}
    for c in df.columns:
        dicc[c] = df[c].values
    return dicc

def save_to_stats_pkl_file(experiment_log_filepath, filename, stats_dict):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "wb") as file_writer:
        pickle.dump(stats_dict, file_writer)


def load_from_stats_pkl_file(experiment_log_filepath, filename):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "rb") as file_reader:
        stats = pickle.load(file_reader)

    return stats


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats
