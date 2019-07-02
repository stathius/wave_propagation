import matplotlib.pyplot as plt
import logging
from utils.arg_extract import get_args
from utils.experiment_runner import ExperimentRunner
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)

if args.continue_experiment:
    experiment.load()
else:
    experiment.create_new()

experiment_runner = ExperimentRunner(experiment)
experiment_runner.run_experiment()
