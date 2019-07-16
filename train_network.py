import matplotlib.pyplot as plt
import logging
from utils.arg_extract import get_args
from utils.experiment_runner import ExperimentRunner
from utils.experiment import Experiment
from utils.experiment_evaluator import evaluate_experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)

if args.continue_experiment:
    experiment.load_from_disk(test=False)
else:
    experiment.create_new()

experiment_runner = ExperimentRunner(experiment)
experiment_runner.run_experiment()

# No need cause it timeouts
# evaluate_experiment(experiment, args)