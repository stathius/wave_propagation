import logging
import matplotlib.pyplot as plt
from utils.arg_extract import get_args
from utils.experiment_evaluator import evaluate_experiment
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)
experiment.load_from_disk(test=True)

evaluate_experiment(experiment, args)