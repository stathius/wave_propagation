import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from models.ConvLSTM import get_convlstm_model
from utils.arg_extract import get_args_train
from utils.experiment_runner import ExperimentRunner
from utils.experiment_setup import ExperimentSetup, get_normalizer, create_new_datasets, create_dataloaders, get_device, save_metadata
from utils.io import save

plt.ioff()
logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args_train()
setup = ExperimentSetup(args.experiment_name)
normalizer = get_normalizer(args.normalizer)
datasets = create_new_datasets(setup.dirs['data'], normalizer)
save(datasets, setup.files['datasets'])
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()

model = get_convlstm_model(args.num_input_frames, args.num_output_frames, args.batch_size, device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

save_metadata(setup.files['metadata'], args, model, optimizer, lr_scheduler, device)

experiment = ExperimentRunner(model=model, lr_scheduler=lr_scheduler,
                              experiment_name=args.experiment_name,
                              num_epochs=args.num_epochs,
                              samples_per_sequence=args.samples_per_sequence,
                              device=device,
                              train_data=data_loaders['train'],
                              val_data=data_loaders['val'],
                              test_data=data_loaders['test'],
                              dirs=setup.dirs,
                              continue_from_epoch=-1,
                              debug=args.debug)
experiment.run_experiment()
