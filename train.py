"""
train.py - training module
"""

# import dependencies
import os
import glob
import argparse
import neptune
import tensorflow as tf
from neptunecontrib.monitoring.keras import NeptuneMonitor

# import local packages
from emorecom.data import Dataset

# read neptune token
with open('neptune_token.txt') as file:
	api_token = file.read()

# initialize neptune logging
neptune.init(
	api_token = api_token,
	project_qualified_name = 'ericngo/emorecom/')
neptune.set_project('ericngo/emorecom')

def train(args):

	# create experiment
	neptune.create_experiment('model') # change to your model name

	# initialize train dataset
	train_path = args.data_path
	
	# initialize train-dataset
	dataset = Dataset(
		data_path = train_path,
		batch_size = 4)
	train_data = dataset(training = True)

	# test train-dataset
	sample = next(iter(train_data))
	print(sample)


	neptune.stop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')
	parser.add_argument('--data-path',
		type = str, default = os.path.join(os.getcwd(), 'dataset', 'train.tfrecords'))
	train(parser.parse_args())
