"""
train.py - training module
"""

# import dependencies
import os
import glob
import argparse
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks, losses, metrics

# import local packages
from emorecom.data import Dataset
from emorecom.model import create_model

# global variables
LOG_DIR = os.path.join(os.getcwd(), 'logs')
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

def train(args):

	# initialize experiment-name
	experiment = 'model-0'

	# initialize train dataset
	train_path = args.data_path
	
	# initialize train-dataset
	print("Creating Data Loading")
	dataset = Dataset(
		data_path = train_path,
		batch_size = 4)
	train_data = dataset(training = True)

	# test train-dataset
	sample = next(iter(train_data))
	print(sample)

	# initialize model
	print("Initialize and compile model")
	MODEL_CONFIGS= {
	}
	model = create_model(configs = MODEL_CONFIGS)

	# set hyperparameters
	PARAMS = {
		'LR' = 0.0001,
		'EPOCH' = 50,
		'CALLBACKS' : [],
		'OPTIMIZER' : optimizers.Adam,
		'LOSS
	}
	# compile model
	OPTIMIZER = optimzers.Adam(learning_rate = PARAMS['LR'])
	LOSS = losses.BinaryCrossEntropy(from_logis = True)
	METRICS = [metrics.accuracy]
	model.compile(optimizer = OPTIMZER, loss = LOSS, metrics = METRICS)

	# set hyperparameters
	print("Start training")
	LOG_DIR = os.path.join(LOG_DIR, experiment)
	CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, experiment)
	CALLBACKS = [
		callbacks.TensorBoard(log_dir = LOG_DIR, write_images = True),
		callbacks.ModelCheckpoint(filepath = CHECKPOINT_PATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')]
	EPOCHS = 50
	STEPS_PER_EPOCH = NONE
	model.fit(train_data, verbose = 1, callbacks = CALLBACKS, epochs = EPOCHS,
		steps_per_epoch = STEPS_PER_EPOCH)

	# save model
	model_path = experiment
	tf.saved_model.save(model_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')
	parser.add_argument('--data-path',
		type = str, default = os.path.join(os.getcwd(), 'dataset', 'train.tfrecords'))
	train(parser.parse_args())
