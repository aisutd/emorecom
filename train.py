"""
train.py - training module
"""

# import dependencies
import os
import glob
import argparse
import tensorflow as tf

# import local packages
from .data import Dataset

def train(args):
	
	# initialize train dataset
	train_image = os.path.join(args.data_path, 'train')
	train_transcript = os.path.join(args.data_path,
		glob.glob('*.json')[0])
	train_label = os.path.join(args.data_path,
		glob.glob('*.csv')[0])
	
	# initialize train-dataset
	dataset = Dataset(
		image = train_image,
		transcript = train_transcript,
		label = train_label,
		batch_size = 1)
	train_data = dataset(training = True)

	# test train-dataset
	sample = next(iter(train_data))
	print(sample)

	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')
	parser.add_argument('--data-path', type = str, default = os.path.join(os.getcwd(), 'warm-up-train'))
	train(parser.parse_args())
