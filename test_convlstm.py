import logging
import matplotlib.pyplot as plt
from models.ConvLSTM import get_convlstm_model
from utils.arg_extract import get_args
from utils.Evaluator import Evaluator
from utils.experiment import Experiment, get_normalizer, load_datasets, create_dataloaders, get_device, load_metadata, load_network, get_transforms
from utils.experiment_runner import test_future_frames
plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()

experiment = Experiment(args.experiment_name)
metadata = load_metadata(experiment.files['metadata'])
normalizer = get_normalizer(args.normalizer_type)
datasets = load_datasets(experiment.files['datasets'])
datasets['Testing data'].root_dir = experiment.dirs['data']
datasets['Testing data'].transform = get_transforms(normalizer)['Test']
dataloaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()

model = get_convlstm_model(metadata['args'].num_input_frames, metadata['args'].num_output_frames, args.num_output_keep_frames, args.batch_size, device)

model = load_network(model, experiment.files['model'])
model.to(device)


logging.info("Start testing")
score_keeper = Evaluator(experiment.dirs['charts'], normalizer)
test_future_frames(model, dataloaders['test'], args.test_starting_point, args.num_total_output_frames, device, score_keeper, experiment.dirs['predictions'], debug=args.debug, normalize=normalizer)
score_keeper.plot(args.show_plots)
score_keeper.plot()
