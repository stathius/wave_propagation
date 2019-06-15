from __future__ import print_function
import logging
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import platform
import time
from utils.Network import Network, MyDataParallel
from utils.Analyser import Analyser
from utils.io import save_network, load_network, save, load, make_folder_results
from utils.WaveDataset import create_datasets, transformVar, normalize
from utils.training import train_epoch, validate, test
from utils.arg_parse import get_args
from utils.Scorekeeper import Scorekeeper

logging.basicConfig(format='%(message)s',level=logging.INFO)

args, device = get_args()

if 'Darwin' in platform.system():
    data_dir = './'
else:
    data_dir = '/disk/scratch/s1680171/wave_propagation/'

if not os.path.isdir("./Results"):
    os.mkdir("./Results")
results_dir = "./Results/" + args.experiment_name 

if not os.path.isdir(results_dir):
    make_folder_results(results_dir)

# Data
filename_data = os.path.join(results_dir,"all_data.pickle")
if os.path.isfile(filename_data):
    logging.info('Loading datasets')
    all_data = load(filename_data)
    train_dataset = all_data["Training data"]
    val_dataset = all_data["Validation data"]
    test_dataset = all_data["Testing data"]
else:
    logging.info('Creating new datasets')
    test_dataset, val_dataset, train_dataset = create_datasets(
         data_dir+"Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False, num_channels=args.num_channels)
    all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
    save(all_data, filename_data)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


# analyser
filename_analyser = os.path.join(results_dir,"analyser.pickle")
if os.path.isfile(filename_analyser):
    logging.info('Loading analyser')
    analyser = load(filename_analyser)
else:
    logging.info('Creating analyser')
    analyser = Analyser(results_dir)

# Model
filename_model = os.path.join(results_dir,"model.pt")
if os.path.isfile(filename_model):
    model = Network(args.num_channels)
    model = load_network(model, filename_model)
else:
    model = Network(args.num_channels)

model.to(device)

optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# Add learning rate schedulers
# Decay LR by a factor of gamma every step_size epochs
scheduler_type = 'plateau'
if scheduler_type == 'step':
    gamma = 0.5
    step_size = 40
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_algorithm, step_size=step_size, gamma=gamma)
elif scheduler_type == 'plateau':
    # Reduce learning rate when a metric has stopped improving
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)

# Save metadata
filename_metadata = os.path.join(results_dir,"metadata.pickle" )
meta_data_dict = {  "args": args,
                    "optimizer": optimizer_algorithm.state_dict(),
                    "scheduler_type": scheduler_type, 
                    "scheduler": lr_scheduler.state_dict()}
save(meta_data_dict, filename_metadata)


if __name__ == "__main__":
    logging.info('Experiment %s' % args.experiment_name)
    logging.info(args)
    logging.info('Start training')
    
    if args.debug:
        epochs=1
    else:
        epochs=50
    for epoch in range(1,epochs+1):
        epoch_start = time.time()

        logging.info('Epoch %d' % epoch)
        train_loss = train_epoch(model, lr_scheduler, epoch, train_dataloader, args.num_input_frames, 
                                args.num_output_frames,args.reinsert_frequency, args.num_channels, device, analyser, plot=args.plot, debug=args.debug)
        analyser.save_epoch_loss(train_loss, epoch)
        validation_loss = validate(model, val_dataloader, args.num_input_frames, args.num_output_frames, args.reinsert_frequency, 
                                    args.num_channels, device, plot=args.plot, debug=args.debug)
        analyser.save_validation_loss(validation_loss, epoch)
        """
        Here we can access analyser.validation_loss to make decisions
        """
        # Learning rate scheduler
        # perform scheduler step if independent from validation loss
        if scheduler_type == 'step':
            lr_scheduler.step()
        # perform scheduler step if dependent on validation loss
        if scheduler_type == 'plateau':
            validation_loss = analyser.validation_loss[-1]
            lr_scheduler.step(validation_loss)
        save_network(model, filename_model)
        save(analyser, filename_analyser)

        epochs_time = time.time() - epoch_start
        logging.info('Epoch %d\tTrain Loss %.6f\tValidation loss: %.6f\tEpoch Time: %.3f' % (epoch, train_loss, validation_loss, epochs_time))

logging.info("Start testing")
score_keeper=Scorekeeper(results_dir, args.num_channels, normalize)
figures_dir = os.path.join(results_dir,'figures')
test(model, test_dataloader, args.test_starting_point, args.num_input_frames, args.reinsert_frequency, 
            args.num_channels, device, score_keeper, figures_dir, plot=True, debug=args.debug, normalize=normalize)

score_keeper.plot()