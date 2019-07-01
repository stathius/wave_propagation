import logging
import matplotlib.pyplot as plt
from models.ConvLSTM import get_convlstm_model
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
from utils.experiment_setup import Experiment, get_normalizer, load_datasets, create_dataloaders, get_device, load_metadata, load_network, get_transforms
from utils.experiment_runner import test_future_frames
plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
# print(args)
setup = Experiment(args.experiment_name)
metadata = load_metadata(setup.files['metadata'])
# print(metadata)
# get normalizer from metadata
normalizer = get_normalizer(args.normalizer_type)
# TODO make this load_dataloaders
datasets = load_datasets(setup.files['datasets'])
datasets['Testing data'].root_dir = setup.dirs['data']
datasets['Testing data'].transform = get_transforms(normalizer)['Test']
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
# up to here
device = get_device()

model = get_convlstm_model(metadata['args'].num_input_frames, metadata['args'].num_output_frames, args.num_autoregress_frames, args.batch_size, device)

model = load_network(model, setup.files['model'])
model.to(device)


logging.info("Start testing")
score_keeper = Scorekeeper(setup.dirs['charts'], normalizer)
test_future_frames(model, data_loaders['test'], args.test_starting_point, args.num_future_test_frames, device, score_keeper, setup.dirs['predictions'], debug=args.debug, normalize=normalizer)
score_keeper.plot(args.show_plots)
score_keeper.plot()
