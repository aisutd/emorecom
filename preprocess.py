"""
preprocess.py - training module
"""

# import dependencies
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from emorecom.utils import regex_replace

# default path
DEFAULT_PATH = os.path.join(os.getcwd(), 'dataset')
EMOTIONS = ['angry', 'disgust', 'feear', 'happy', 'sad', 'surprise', 'neutral', 'others']

def train_concat(file_name, image_path, transcripts, labels, indices):
	"""
	train_concat - function to concat images, transcripts, and labels together
	Inputs:
		- file_name : str
		- image_path : str
		- transcripts : str
		- labels : lits of int
		- indices : list of int
	"""

	def parse(image, transcript, label):
		"""
		parse - function to parse image, transcript, and label to tf.train.Example
		Inputs:
			- image : str
			- transcript : str
			- label : str
		Outputs:
			- _ : tf.train.Example
		"""
		output = {
			'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
			'transcripts' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [transcript])),
			'label' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [label]))}
		
		return tf.train.Example(features = tf.train.Features(feature = output)).SerializeToString()

	with tf.io.TFRecordWriter(file_name) as writer:
		for idx in indices:
			transcript = transcripts[idx]
			try:
				print("Processing {} sample".format(idx))

				# retrieve labels
				label = list(
					labels[labels['image_id'] == transcript['img_id']]
						.iloc[0][EMOTIONS])
				label = ','.join([str(x) for x in label]).encode('utf-8') # convert to string

				# retrieve transcripts
				transcript['dialog'] = [x for x in transcript['dialog'] if isinstance(x, str)] # remove nan transcript
				texts = ';'.join(transcript['dialog']) if len(transcript['dialog']) > 0 else ''
				texts = texts.encode('utf-8') # encode to butes

				# retrieve image
				img = os.path.join(image_path, transcript['img_id'] + '.jpg').encode('utf-8')

				# parse image, transcript, label to tfrecord-example
				writer.write(parse(img, texts, label))

			except Exception as e:
				print(e)

def test_concat(file_name, image_path, transcripts):
	"""
	test_concat - function to concat images and transcripts together
	Inputs:
		- file_name : str
		- image_path : str
		- transcript : str
 	"""

	def parse(image, transcript):
		"""
		parse - function to parse image and transcript into tf.train.Example
		Inputs:
			- image : str
			- transcript : str
		Outputs:
			- _ : tf.train.Example
		"""
		output = {
			'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
			'transcripts' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [transcript]))
		}
		return tf.train.Example(features = tf.train.Features(feature = output)).SerializeToString()

	with tf.io.TFRecordWriter(file_name) as writer:
		for transcript, idx in zip(transcripts, range(len(transcripts))):
			try:
				print("Processing {} sample".format(idx))

				# retrieve transcripts
				transcript['dialog'] = [x for x in transcript['dialog'] if isinstance(x, str)] #remove nan text
				texts = ';'.join(transcript['dialog']) if len(transcript['dialog']) > 0 else ''
				texts = texts.encode('utf-8')
			
				# retrieve image
				img = os.path.join(image_path, transcript['img_id'] + '.jpg').encode('utf-8')

				# parse image, transcript, label to tfrecord-example
				writer.write(parse(img, texts))

			except Exception as e:
				print(e)

def test(filename):
	"""
	test - function to inspect if data is concatonated correctly
	"""

	# read tfrecord file
	data = tf.data.TFRecordDataset(filename)

	for sample in data.take(5):
		print(sample)

	def _parse(input):
		feature_details = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string),
			'label' : tf.io.FixedLenFeature([], tf.string)}
		return tf.io.parse_single_example(input, feature_details)

	#parsed_data = data.map(lambda x: tf.io.parse_single_example(x, feature_details))
	parsed_data = data.map(_parse)
	print(next(iter(parsed_data)))

def build_vocab(inputs, vocab_name):
	print("Build vocabs")

	vocabs = [] # initialize empty list of vocabs

	for sent in inputs:
		# regex replace
		sent = tf.constant(sent)
		sent = regex_replace(sent).numpy().decode('utf-8')

		# remove trivial whitepsaces
		sent = re.sub("\s+", " ", sent)

		# add to vocabs
		vocabs.extend(sent.split())

	# find unique words and sort alphabetically
	vocabs = list(set(vocabs))

	# add special tokens (similar to BERT WordPiece Tokenizer)
	vocabs.extend(['[PAD]', '[SEP]'])
	vocabs = ['[UNK]'] + vocabs


	# write voacbs file
	with open(vocab_name, 'w') as file:
		for vocab in vocabs:
			file.write('%s\n' % vocab)

def main(args):
	
	# initialize train dataset
	transcripts = os.path.join(DEFAULT_PATH, args.transcript)
	image_path = os.path.join(DEFAULT_PATH, args.image)

	# read transcripts
	with open(transcripts) as file:
		transcripts = json.load(file)

	# read labels
	if args.label:
		labels = os.path.join(DEFAULT_PATH, args.label)	
		labels = pd.read_csv(labels)

		# rename columns
		labels = labels.rename(
			columns = {
				old:new for old, new in zip(labels.columns, ['id', 'image_id'] + EMOTIONS)})

	# concat images, transcripts, and labels (if training is True)
	if args.training:
		# check if given args.label is valid
		assert args.label, "Training modee requires valid labels"

		print("Concat images, transcripts, and labels")

		# generate retrivial indicees for transcripts
		indices = np.arange(start = 0, stop = len(transcripts))

		if args.test_size > 0.0:
			# split indices
			train_indices, val_indices = train_test_split(indices, test_size = args.test_size, random_state = 2021)
			# training
			print("Concat training data")
			train_output = os.path.join(DEFAULT_PATH, args.output)
			train_concat(train_output, image_path, transcripts, labels, train_indices)
			test(train_output)

			# val
			print("Concat validation data")
			val_output = os.path.join(DEFAULT_PATH, args.val_output)
			train_concat(val_output, image_path, transcripts, labels, val_indices)
			test(val_output)
		else:
			print("Concat training data")
			output = os.path.join(DEFAULT_PATH, args.output)
			train_concat(output, image_path, transcripts, labels, indices)
			test(output)

		# build vocabs
		## flatten transcripts
		transcripts = [item for sublist in transcripts for item in sublist['dialog']]
	
		## retrieve vocabs
		build_vocab(inputs = transcripts,
			vocab_name = os.path.join(DEFAULT_PATH, args.vocab_name))

	else:
		print("Concat data for inference")
		output = os.path.join(DEFAULT_PATH, args.output)
		test_concat(output, image_path, transcripts)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')

	# add arguments
	parser.add_argument('--training', default = False, action = 'store_true')
	parser.add_argument('--image', type = str, default = os.path.join('warm-up-train', 'train'))
	parser.add_argument('--transcript', type = str, default = os.path.join('warm-up-train', 'train_transcriptions.json'))
	parser.add_argument('--label', type = str, default = os.path.join('warm-up-train', 'train_emotion_labels.csv'))
	parser.add_argument('--test-size', type = float, default = 0.0)
	parser.add_argument('--output', type = str, default = 'train.tfrecords')
	parser.add_argument('--val-output', type = str, default = 'val.tfrecords')
	parser.add_argument('--vocab-name', type =str, default = 'vocabs.txt')
	main(parser.parse_args())
