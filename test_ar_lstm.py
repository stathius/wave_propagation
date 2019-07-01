import logging
import torch
import matplotlib.pyplot as plt
from models.AR_LSTM import AR_LSTM, test_ar_lstm
from utils.arg_extract import get_args_test
from utils.Scorekeeper import Scorekeeper
from utils.experiment import Experiment, get_device, load_metadata, load_datasets, load_network, create_dataloaders, get_normalizer, get_transforms
plt.ioff()

logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args_test()
print(args)
experiment = Experiment(args.experiment_name)
metadata = load_metadata(experiment.files['metadata'])
print(metadata)
# if metadata['args'].normalizer
normalizer = get_normalizer(args.normalizer_type)
datasets = load_datasets(experiment.files['datasets'])
datasets['Testing data'].root_dir = experiment.dirs['data']
datasets['Testing data'].transform = get_transforms(normalizer)['Test']
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()

model = AR_LSTM(metadata['args'].num_input_frames, args.reinsert_frequency, device)
model = load_network(model, experiment.files['model'])
model.to(device)

score_keeper = Scorekeeper(experiment.dirs['charts'], normalizer)

test_ar_lstm(model, data_loaders['test'], args.test_starting_point, args.num_input_frames, args.reinsert_frequency, device, score_keeper, experiment.dirs['predictions'], args.debug, normalizer)
score_keeper.plot()
