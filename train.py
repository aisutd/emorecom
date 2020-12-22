"""
train.py - training module
"""

# import dependencies
import os
import glob
import argparse
import tensorflow as tf

# import local packages
from emorecom.data import Dataset

def train(args):

	# initialize experiment-name
	experiment = 'model-0'

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')
	parser.add_argument('--data-path',
		type = str, default = os.path.join(os.getcwd(), 'dataset', 'train.tfrecords'))
	train(parser.parse_args())
