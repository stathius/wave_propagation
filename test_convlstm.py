import logging
import matplotlib.pyplot as plt
from utils.arg_extract import get_args
from utils.experiment_evaluator import Evaluator, test_future_frames
from utils.experiment import Experiment

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)
experiment.load_from_disk()

evaluator = Evaluator(experiment.args.test_starting_point, args.num_total_output_frames, experiment.normalizer)


logging.info("Start testing")
test_future_frames(experiment.model, experiment.dataloaders['test'], experiment.args.test_starting_point, args.num_total_output_frames, experiment.device, evaluator, experiment.dirs['predictions'], debug=args.debug, normalize=experiment.normalizer)
# score_keeper.plot(args.show_plots)
# score_keeper.plot()
