from __future__ import print_function
import logging
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import os
import platform
import time
from models.AR_LSTM import AR_LSTM, train_epoch, validate, test
from utils.Analyser import Analyser
from utils.io import save_network, save, create_results_folder, save_datasets_to_file
from utils.WaveDataset import create_datasets, transformVar, normalize, create_dataloaders
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


dirs = create_results_folder(base_folder=base_folder, experiment_name=args.experiment_name)

logging.info('Creating datasets')
train_dataset, val_dataset, test_dataset = create_datasets(os.path.join(data_dir, "Video_Data/"), transformVar, test_fraction=0.15, validation_fraction=0.15)
filename_data = os.path.join(dirs['results'], "all_data.pickle")
save_datasets_to_file(train_dataset, val_dataset, test_dataset, filename_data)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers)

# analyser
filename_analyser = os.path.join(dirs['pickles'], "analyser.pickle")
logging.info('Creating analyser')
analyser = Analyser(dirs['results'])

# Model
filename_model = os.path.join(dirs['models'], "model.pt")
model = AR_LSTM(args.num_input_frames, args.num_output_frames, device)
model.to(device)

optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)

# Save metadata
filename_metadata = os.path.join(dirs['pickles'], "metadata.pickle" )
meta_data_dict = {  "args": args, "optimizer": optimizer_algorithm.state_dict(), "scheduler": lr_scheduler.state_dict(), "model": "%s" % model}
save(meta_data_dict, filename_metadata)

if __name__ == "__main__":
    logging.info('Experiment %s' % args.experiment_name)
    logging.info(meta_data_dict)
    logging.info('Start training')

    if args.debug:
        epochs=1
    else:
        epochs=50
    for epoch in range(1,epochs+1):
        epoch_start = time.time()

        logging.info('Epoch %d' % epoch)
        train_loss = train_epoch(model, lr_scheduler, epoch, train_dataloader, args.num_input_frames,
                                args.num_output_frames,args.reinsert_frequency, device, analyser, show_plots=args.show_plots, debug=args.debug)
        analyser.save_epoch_loss(train_loss, epoch)
        validation_loss = validate(model, val_dataloader, args.num_input_frames, args.num_output_frames, args.reinsert_frequency,
                                   device, show_plots=args.show_plots, debug=args.debug)
        analyser.save_validation_loss(validation_loss, epoch)
        validation_loss = analyser.validation_loss[-1]
        lr_scheduler.step(validation_loss)
        save_network(model, filename_model)
        save(analyser, filename_analyser)

        epochs_time = time.time() - epoch_start
        logging.info('Epoch %d\tTrain Loss %.6f\tValidation loss: %.6f\tEpoch Time: %.3f' % (epoch, train_loss, validation_loss, epochs_time))

logging.info("Start testing")
score_keeper=Scorekeeper(dirs['charts'], normalize)
figures_dir = os.path.join(dirs['results'],'test_charts')
test(model, test_dataloader, args.test_starting_point, args.num_input_frames, args.reinsert_frequency,
            device, score_keeper, dirs['predictions'], show_plots=args.show_plots, debug=args.debug, normalize=normalize)
score_keeper.plot(args.show_plots)