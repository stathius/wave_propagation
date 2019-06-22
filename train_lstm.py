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
from utils.Network import Network
from utils.Analyser import Analyser
from utils.io import save_network, load_network, save, load, create_results_folder
from utils.WaveDataset import create_datasets, transformVar, normalize
from utils.training import train_epoch, validate, test
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
# import matplotlib
# matplotlib.use('Agg') # don't allow showing plots
import matplotlib.pyplot as plt
plt.ioff()


logging.basicConfig(format='%(message)s',level=logging.INFO)

args, device = get_args()

# Data
if 'Darwin' in platform.system():
    base_folder = '/Users/stathis/Code/thesis/wave_propagation/'
    data_dir = base_folder
else:
    base_folder = '/home/s1680171/wave_propagation/'
    data_dir = '/disk/scratch/s1680171/wave_propagation/'
    

results_dir = create_results_folder(base_folder=base_folder, experiment_name=args.experiment_name)

logging.info('Creating new datasets')
test_dataset, val_dataset, train_dataset = create_datasets(os.path.join(data_dir, "Video_Data/"), transformVar, 
                                                            test_fraction=0.15, validation_fraction=0.15)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

filename_data = os.path.join(results_dir,"all_data.pickle")
all_data = {"Training data": train_dataset, "Validation data": val_dataset, "Testing data": test_dataset}
save(all_data, filename_data)


# analyser
filename_analyser = os.path.join(results_dir,"analyser.pickle")
logging.info('Creating analyser')
analyser = Analyser(results_dir)

# Model
filename_model = os.path.join(results_dir,"model.pt")
model = Network(args.num_channels, device)
model.to(device)

optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)

# Save metadata
filename_metadata = os.path.join(results_dir,"metadata.pickle" )
meta_data_dict = {  "args": args, "optimizer": optimizer_algorithm.state_dict(), "scheduler": lr_scheduler.state_dict()}
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
                                args.num_output_frames,args.reinsert_frequency, args.num_channels, device, analyser, show_plots=args.show_plots, debug=args.debug)
        analyser.save_epoch_loss(train_loss, epoch)
        validation_loss = validate(model, val_dataloader, args.num_input_frames, args.num_output_frames, args.reinsert_frequency, 
                                    args.num_channels, device, show_plots=args.show_plots, debug=args.debug)
        analyser.save_validation_loss(validation_loss, epoch)
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
            args.num_channels, device, score_keeper, figures_dir, show_plots=args.show_plots, debug=args.debug, normalize=normalize)
score_keeper.plot(args.show_plots)