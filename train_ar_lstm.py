from __future__ import print_function
import logging
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.AR_LSTM import AR_LSTM, run_iteration
from utils.Analyser import Analyser
from utils.io import save, save_json
from utils.arg_extract import get_args
from utils.Scorekeeper import Scorekeeper
from utils.experiment_setup import Experiment, get_normalizer, create_new_datasets, create_dataloaders, get_device, save_metadata, save_network
from utils.experiment_runner import test_future_frames
plt.ioff()

logging.basicConfig(format='%(message)s', level=logging.INFO)

args = get_args()
setup = Experiment(args.experiment_name)
normalizer = get_normalizer(args.normalizer_type)
datasets = create_new_datasets(setup.dirs['data'], normalizer)
save(datasets, setup.files['datasets'])
data_loaders = create_dataloaders(datasets, args.batch_size, args.num_workers)
device = get_device()

analyser = Analyser(setup.dirs['pickles'])

model = AR_LSTM(args.num_input_frames, args.reinsert_frequency, device)
model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

save_metadata(setup.files['metadata'], args, model, optimizer, lr_scheduler, device)

logging.info('Start training')
for epoch in range(1, args.num_epochs+1):
    epoch_start = time.time()

    logging.info('Epoch %d' % epoch)
    train_loss = run_iteration(model, lr_scheduler, epoch, data_loaders['train'], args.num_input_frames, args.num_output_frames, args.reinsert_frequency, device, analyser, training=True, debug=args.debug)
    analyser.save_epoch_loss(train_loss, epoch)
    with torch.no_grad():
        validation_loss = run_iteration(model, lr_scheduler, epoch, data_loaders['val'], args.num_input_frames, args.num_output_frames, args.reinsert_frequency, device, analyser, training=False, debug=args.debug)
    analyser.save_validation_loss(validation_loss, epoch)
    validation_loss = analyser.validation_loss[-1]
    lr_scheduler.step(validation_loss)
    save_network(model, setup.files['model'])
    save(analyser, setup.files['analyser'])

    epochs_time = time.time() - epoch_start
    logging.info('Epoch %d\tTrain Loss %.6f\tValidation loss: %.6f\tEpoch Time: %.3f' % (epoch, train_loss, validation_loss, epochs_time))

logging.info("Start testing")
score_keeper = Scorekeeper(setup.dirs['charts'], normalizer)
test_future_frames(model, data_loaders['test'], args.test_starting_point, args.num_future_test_frames, device, score_keeper, setup.dirs['predictions'], debug=args.debug, normalize=normalizer)
score_keeper.plot()
