import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.io import save_figure, save_json, load_json
import os
import time


class Logger():
    """
    Saves network data for later analasys. Epochwise loss, Batchwise loss, Accuracy (not currently in use) and
    Validation loss
    """
    def __init__(self):
        self.logs = {'train_loss': [],
                     'validation_loss': [],
                     'epoch_nr': [],
                     'batch_loss': [],
                     'batch_nr': []
                     }
        self.start_time = time.time()

    def record_epoch_losses(self, train_loss, val_loss, epoch):
        """
        Creates two lists, one of losses and one of index of epoch
        """
        self.logs['train_loss'].append(train_loss)
        self.logs['validation_loss'].append(val_loss)
        self.logs['epoch_nr'].append(epoch)

    def record_loss_batchwise(self, loss, batch_increment=1):
        """
        Creates two lists, one of losses and one of index of batch
        """
        if len(self.logs['batch_nr']) > 0:
            batch_num = self.logs['batch_nr'][-1] + batch_increment  # increase by one
        else:
            batch_num = 0
        self.logs['batch_loss'].append(loss)
        self.logs['batch_nr'].append(batch_num)

    def get_best_val_loss(self):
        return min(self.logs['validation_loss'])

    def get_current_epoch_loss(self, type):
        return self.logs['%s_loss' % type][-1]

    def get_best_epoch(self):
        return np.argmin(self.logs['validation_loss'])

    def get_last_epoch(self):
        return self.logs['epoch_nr'][-1]

    def plot_batchwise_loss(self):
        fig = plt.figure().add_axes()
        ax = plt.gca()
        ax.set(yscale='log')
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Batch": self.logs['batch_nr'], "Loss": self.logs['batch_loss']})
        sns.lineplot(x="Batch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        return fig

    def plot_validation_loss(self, title=None):
        hue = []
        loss = []
        nr = []
        for i, element in enumerate(self.logs['train_loss']):
            loss.append(element)
            nr.append(self.logs['epoch_nr'][i])
            hue.append("Training")
        for i, element in enumerate(self.logs['validation_loss']):
            loss.append(element)
            nr.append(self.logs['epoch_nr'][i])
            hue.append("Validation")
        fig = plt.figure().add_axes()
        ax = plt.gca()
        ax.set(yscale='log')
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": nr, "Loss": loss, "Dataset": hue})
        sns.lineplot(x="Epoch", y="Loss", hue="Dataset", data=pd.DataFrame.from_dict(data), ax=fig)
        if title is not None:
            ax.set_title(title)
        return fig

    def save_batchwise_loss(self, figures_dir):
        fig = self.batchwise_loss_plot()
        save_figure(os.path.join(figures_dir, "Batch_Loss"), obj=fig)
        plt.close()

    def save_validation_loss_plot(self, figures_dir):
        fig = self.plot_validation_loss()
        save_figure(os.path.join(figures_dir, "Validation_Loss"), obj=fig)
        plt.close()

    def save_training_progress(self, file):
        progress = {'latest_epoch': self.get_last_epoch(),
                    'best_val_loss': self.get_best_val_loss(),
                    'best_epoch': self.get_best_epoch(),
                    'time': time.time() - self.start_time
                    }
        save_json(progress, file)

    def load_from_json(self, filename):
        self.logs = load_json(filename)

    def save_to_json(self, filename):
        save_json(self.logs, filename)
