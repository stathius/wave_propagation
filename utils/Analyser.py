import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.io import figure_save
import os

class Analyser():
    """
    Saves network data for later analasys. Epochwise loss, Batchwise loss, Accuracy (not currently in use) and
    Validation loss
    """
    def __init__(self, results_dir):
        self.epoch_loss = []
        self.epoch_nr = []
        self.batch_loss = []
        self.batch_nr = []
        self.accuracy = []
        self.epoch_acc = []
        self.validation_loss = []
        self.validation_nr = []
        self.results_dir = results_dir
        self.figures_dir = os.join.path(results_dir, 'figures')

    def save_epoch_loss(self, loss, epoch):
        """
        Creates two lists, one of losses and one of index of epoch
        """
        self.epoch_loss.append(loss)
        self.epoch_nr.append(epoch)

    def plot_loss(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": self.epoch_nr, "Loss": self.epoch_loss})
        sns.lineplot(x="Epoch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(os.path.join(self.figures_dir, "Epoch_Loss"), obj=fig)
        plt.show()

    def save_loss_batchwise(self, loss, batch_increment=1):
        """
        Creates two lists, one of losses and one of index of batch
        """
        self.batch_loss.append(loss)
        self.batch_nr.append(self.batch_nr[len(self.batch_nr) - 1] + batch_increment) if len(self.batch_nr) else self.batch_nr.append(batch_increment)

    def plot_loss_batchwise(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Batch": self.batch_nr, "Loss": self.batch_loss})
        sns.lineplot(x="Batch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(os.path.join(self.figures_dir, "Batch_Loss"), obj=fig)
        plt.show()

    def save_validation_loss(self, loss, epoch):
        """
        Creates two lists, one of validation losses and one of index of epoch
        """
        self.validation_loss.append(loss)
        self.validation_nr.append(epoch)

    def plot_validation_loss(self):
        """
        Plots validation and epoch loss next to each other
        """
        hue = []
        loss = []
        nr = []
        for i, element in enumerate(self.epoch_loss):
            loss.append(element)
            nr.append(self.epoch_nr[i])
            hue.append("Training")
        for i, element in enumerate(self.validation_loss):
            loss.append(element)
            nr.append(self.validation_nr[i])
            hue.append("Validation")
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": nr, "Loss": loss, "Dataset": hue})
        sns.lineplot(x="Epoch", y="Loss", hue="Dataset", data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(os.path.join(self.figures_dir, "Validation_Loss"), obj=fig)
        plt.show()
        return data

        # df_1['region'] = 'A'
        # df_2['region'] = 'B'
        # df_3['region'] = 'C'
        # df = pd.concat([df_1,df_2,df_3])
        # sns.pointplot(ax=ax,x=x_col,y=y_col,data=df,hue='region')

