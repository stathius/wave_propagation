import matplotlib.pyplot as plt
import logging
from utils.arg_extract import get_args
from utils.experiment_evaluator import Evaluator, get_sample_predictions
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)
experiment.load_from_disk()

evaluator = Evaluator(args.test_starting_point, args. num_total_output_frames, experiment.normalizer)

evaluator.get_experiment_metrics(experiment)
evaluator.save_plots(experiment.dirs['charts'])
evaluator.save(experiment.files['evaluator'])
