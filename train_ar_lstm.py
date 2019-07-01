import logging
import time
import torch
import matplotlib.pyplot as plt
from models.AR_LSTM import run_iteration
from utils.Analyser import Analyser
from utils.io import save
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
from utils.experiment import Experiment, save_network
from utils.experiment_runner import test_future_frames
plt.ioff()

logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
experiment = Experiment(args)
analyser = Analyser(experiment.dirs['pickles'])

if args.continue_experiment:
    experiment.load()
else:
    experiment.create_new()


logging.info('Start training')
for epoch in range(1, args.num_epochs+1):
    epoch_start = time.time()

    logging.info('Epoch %d' % epoch)
    train_loss = run_iteration(experiment.model, experiment.lr_scheduler, epoch, experiment.dataloaders['train'], args.num_input_frames, args.num_output_frames, args.reinsert_frequency, experiment.device, analyser, training=True, debug=args.debug)
    analyser.save_epoch_loss(train_loss, epoch)
    with torch.no_grad():
        validation_loss = run_iteration(experiment.model, experiment.lr_scheduler, epoch, experiment.dataloaders['val'], args.num_input_frames, args.num_output_frames, args.reinsert_frequency, experiment.device, analyser, training=False, debug=args.debug)
    analyser.save_validation_loss(validation_loss, epoch)
    validation_loss = analyser.validation_loss[-1]
    experiment.lr_scheduler.step(validation_loss)
    save_network(experiment.model, experiment.files['model'])
    save(analyser, experiment.files['analyser'])

    epochs_time = time.time() - epoch_start
    logging.info('Epoch %d\tTrain Loss %.6f\tValidation loss: %.6f\tEpoch Time: %.3f' % (epoch, train_loss, validation_loss, epochs_time))

logging.info("Start testing")
score_keeper = Scorekeeper(experiment.dirs['charts'], experiment.normalizer)
test_future_frames(experiment.model, experiment.dataloaders['test'], args.test_starting_point, args.num_future_test_frames, experiment.device, score_keeper, experiment.dirs['predictions'], debug=args.debug, normalize=experiment.normalizer)
score_keeper.plot()
