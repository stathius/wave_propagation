import logging
import matplotlib.pyplot as plt
from utils.arg_extract import get_args
from utils.experiment_evaluator import Evaluator, get_sample_predictions
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
args.num_workers = 4
args.batch_size = 16
experiment = Experiment(args)
experiment.load_from_disk(test=True)

evaluator = Evaluator(args.test_starting_point, experiment.normalizer)

logging.info("Start testing")
evaluator.compute_experiment_metrics(experiment, args.num_total_output_frames, debug=args.debug)
evaluator.save_metrics_plots(experiment.dirs['charts'])
evaluator.save_to_file(experiment.files['evaluator'] % args.test_starting_point)
# Get the sample plots after you compute everything else because the dataloader iterates from the beginning
logging.info("Generate prediction plots")
get_sample_predictions(experiment.model, experiment.dataloaders['test'], experiment.device, experiment.dirs['predictions'], experiment.normalizer, args.debug)
