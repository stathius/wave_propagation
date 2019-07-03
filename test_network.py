import logging
import matplotlib.pyplot as plt
from utils.arg_extract import get_args
from utils.experiment_evaluator import Evaluator, get_sample_predictions
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)
experiment.load_from_disk()

evaluator = Evaluator(args.test_starting_point, args.num_total_output_frames, experiment.normalizer,)

logging.info("Generate prediction plots")
get_sample_predictions(experiment.model, experiment.dataloaders['test'], experiment.device, experiment.dirs['predictions'], experiment.normalizer, args.debug)
logging.info("Start testing")
evaluator.compute_experiment_metrics(experiment, debug=args.debug)
evaluator.save_metrics_plots(experiment.dirs['charts'])
evaluator.save_to_file(experiment.files['evaluator'])
