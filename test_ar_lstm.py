import logging
import torch
import matplotlib.pyplot as plt
from models.AR_LSTM import AR_LSTM, test_ar_lstm
from utils.arg_extract import get_args_test
from utils.Scorekeeper import Scorekeeper
from utils.experiment_setup import ExperimentSetup, get_device, load_metadata, load_datasets, load_network, create_dataloaders, get_normalizer
plt.ioff()

logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args_test()
print(args)
setup = ExperimentSetup(args.experiment_name)
metadata = load_metadata(setup.files['metadata'])
print(metadata)
# if metadata['args'].normalizer
normalizer = get_normalizer(args.normalizer_type)
datasets = load_datasets(setup.files['datasets'])
datasets['Testing data'].root_dir = setup.dirs['data']
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()

model = AR_LSTM(num_input_frames=5, num_output_frames=1, device=device)
model = load_network(model, setup.files['model'])
model.to(device)

score_keeper = Scorekeeper(setup.dirs['charts'], normalizer)

with torch.no_grad():
    test_ar_lstm(model, data_loaders['test'], args.test_starting_point, args.num_input_frames, args.reinsert_frequency, device, score_keeper, setup.dirs['predictions'], show_plots=args.show_plots, debug=args.debug, normalize=normalizer)
score_keeper.plot(show_plots=args.show_plots)
