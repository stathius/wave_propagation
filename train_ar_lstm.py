from __future__ import print_function
import logging
import os
import time
import torch.optim as optim
from models.AR_LSTM import AR_LSTM, train_epoch, validate, test
from utils.Analyser import Analyser
from utils.io import save_network, save
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
from utils.experiment_setup import ExperimentSetup
import matplotlib.pyplot as plt
plt.ioff()

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()
logging.info('Experiment %s' % args.experiment_name)

setup = ExperimentSetup(args)
dirs = setup.get_dirs()
data_loaders, normalizer = setup.get_dataloaders()
device = setup.get_device()


analyser = Analyser(dirs['results'])
filename_analyser = os.path.join(dirs['pickles'], "analyser.pickle")
filename_model = os.path.join(dirs['models'], "model.pt")

model = AR_LSTM(args.num_input_frames, args.num_output_frames, device)
model.to(device)
logging.info('Start training')

# Optimizer and scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

# Saving metadata of experiment
setup.save_metadata(model, optimizer, lr_scheduler, device)

for epoch in range(1, args.num_epochs+1):
    epoch_start = time.time()

    logging.info('Epoch %d' % epoch)
    train_loss = train_epoch(model, lr_scheduler, epoch, data_loaders['train'], args.num_input_frames, args.num_output_frames,args.reinsert_frequency, device, analyser, show_plots=args.show_plots, debug=args.debug)
    analyser.save_epoch_loss(train_loss, epoch)
    validation_loss = validate(model, data_loaders['val'], args.num_input_frames, args.num_output_frames, args.reinsert_frequency, device, show_plots=args.show_plots, debug=args.debug)
    analyser.save_validation_loss(validation_loss, epoch)
    validation_loss = analyser.validation_loss[-1]
    lr_scheduler.step(validation_loss)
    save_network(model, filename_model)
    save(analyser, filename_analyser)

    epochs_time = time.time() - epoch_start
    logging.info('Epoch %d\tTrain Loss %.6f\tValidation loss: %.6f\tEpoch Time: %.3f' % (epoch, train_loss, validation_loss, epochs_time))

logging.info("Start testing")
score_keeper = Scorekeeper(dirs['charts'], normalizer)
figures_dir = os.path.join(dirs['results'], 'test_charts')
test(model, data_loaders['test'], args.test_starting_point, args.num_input_frames, args.reinsert_frequency, device, score_keeper, dirs['predictions'], show_plots=args.show_plots, debug=args.debug, normalize=normalizer)
score_keeper.plot(args.show_plots)
