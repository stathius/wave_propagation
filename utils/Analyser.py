import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.io import figure_save

class Analyser():
    """
    Saves network data for later analasys. Epochwise loss, Batchwise loss, Accuracy (not currently in use) and
    Validation loss
    """
    def __init__(self, maindir1):
        self.epoch_loss = []
        self.epoch_nr = []
        self.batch_loss = []
        self.batch_nr = []
        self.accuracy = []
        self.epoch_acc = []
        self.validation_loss = []
        self.validation_nr = []
        self.maindir1 = maindir1

    def save_loss(self, loss, epoch_increment=1):
        """
        Creates two lists, one of losses and one of index of epoch
        """
        self.epoch_loss.append(loss)
        self.epoch_nr.append(self.epoch_nr[len(self.epoch_nr) - 1] + epoch_increment) if len(self.epoch_nr)            else self.epoch_nr.append(epoch_increment)

    def plot_loss(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": self.epoch_nr, "Loss": self.epoch_loss})
        sns.lineplot(x="Epoch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.maindir1 + "Epoch_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_loss_batchwise(self, loss, batch_increment=1):
        """
        Creates two lists, one of losses and one of index of batch
        """
        self.batch_loss.append(loss)
        self.batch_nr.append(self.batch_nr[len(self.batch_nr) - 1] + batch_increment) if len(self.batch_nr)            else self.batch_nr.append(batch_increment)

    def plot_loss_batchwise(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Batch": self.batch_nr, "Loss": self.batch_loss})
        sns.lineplot(x="Batch", y="Loss",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.maindir1 + "Batch_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_accuracy(self, accuracy, epoch_increment=1):
        """
        Creates two lists, one of accuracy and one of index of the accuracy (batchwise or epochwise)
        NOT IN USE
        """
        self.accuracy.append(accuracy)
        self.epoch_acc.append(self.epoch_acc[len(self.epoch_acc) - 1] + epoch_increment) if len(self.epoch_acc)            else self.epoch_acc.append(epoch_increment)

    def plot_accuracy(self):
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": self.epoch_acc, "Accuracy": self.accuracy})
        sns.lineplot(x="Epoch", y="Accuracy",
                     data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.maindir1 + "Accuracy" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()

    def save_validation_loss(self, loss, epoch_increment=1):
        """
        Creates two lists, one of validation losses and one of index of epoch
        """
        self.validation_loss.append(loss)
        self.validation_nr.append(self.validation_nr[len(self.validation_nr) - 1] + epoch_increment) if             len(self.validation_nr) else self.validation_nr.append(epoch_increment)

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
            hue.append("Training Loss")
        for i, element in enumerate(self.validation_loss):
            loss.append(element)
            nr.append(self.validation_nr[i])
            hue.append("Validation Loss")
        fig = plt.figure().add_axes()
        sns.set(style="darkgrid")  # darkgrid, whitegrid, dark, white, and ticks
        sns.set_context("talk")
        data = {}
        data.update({"Epoch": nr, "Loss": loss, "hue": hue})
        sns.lineplot(x="Epoch", y="Loss", hue="hue", data=pd.DataFrame.from_dict(data), ax=fig)
        figure_save(self.maindir1 + "Validation_Loss" + Type_Network + "_Project_v%03d" % version, obj=fig)
        plt.show()
