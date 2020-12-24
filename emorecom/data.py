"""
data.py - data-loading module
"""

# import dependencies
import os
import cv2
import pickle
import tensorflow as tf

# import local packages
from emorecom.utils import basic_text_proc

class Dataset:
	"""
	Dataset - class to implement Tensorflow Data API
	"""

	def __init__(self, data_path, vocabs, batch_size = 1, buffer_size = 512, seed = 2021):
		"""
		Class constructor:
		Inputs:
			- data_path : str or list of str
				Path(s) to TFRecord dataset
			- vocabs : str
				Path to vocabs dictionary
			- batch_size : integer
				Number of smapels/batch
			- buffer_size : integer
				Number of samples for shuffling
			- seed : integer
				Random seed
		"""
		# set tensorflow random seed
		tf.random.set_seed(seed)

		# parse arguments
		self.data = tf.data.TFRecordDataset(data_path).cache() # cache data
		self.batch_size = batch_size
		self.buffer_size = buffer_size

		# read vocabs dictionary
		self.vocabs = self.load_vocabs(vocabs)

		# initialize global feature dictionary
		self.train_features = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string),
			'label' : tf.io.FixedLenFeature([], tf.string)}

		self.test_features = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string)}

	def load_vocabs(self, file):
		"""
		load_vocabs - function to load vocabularies
		Inputs:
			- file : str
				Path to  vocabulary dictionary
		"""
		with open(file, 'rb') as file:
			vocabs = pickle.load(file)

	def parse_train(self):
		"""
		parse_train - function to parse training data in TFRecord format
		"""
		# shuffle data
		data = self.data.shuffle(buffer_size = self.buffer_size)

		# read data
		@tf.function
		def _parse(example):
			example = tf.io.parse_single_example(example, self.train_features)

			# read image
			example['image'] = tf.io.read_file(example['image'])

			return {'image' : example['image'], 'transcripts' : example['transcripts'], 'label' : example['label']}
		data = data.cache().map(_parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)

		return data

	def parse_test(self):
		"""
		parse_test - function to parse testing data in TFRecord format
		"""

		# read data
		@tf.function
		def _parse(example):
			example = tf.io.parse_single_example(example, self.test_features)
			return {'image' : example['image'], 'transcripts' : example['transcripts']}

		return data

	@tf.function
	def _image(self, input):
		"""
		process image
		"""
		return input

	@tf.function
	def _transcripts(self, input):
		"""
		process transcripts
		"""
		return basic_text_proc(input)

	@tf.function
	def _label(self, input):
		"""
		process label
		"""

		# split lable by ','
		input = tf.strings.split(input, sep = ',')

		# convert str to integer
		input = tf.strings.to_number(input, out_type = tf.int32)
		return input

	@tf.function
	def process_train(self, sample):
		"""
		process_train - function to preprocess image, text, and label
		"""
		return self._image(sample['image']), self._transcripts(sample['transcripts']),self._label(sample['label'])

	@tf.function
	def process_test(self, sample):
		"""
		process_test - function to preprocess image and text only
		"""

		return self._image(sample['image']), self._transcripts(sample['transcripts'])

	def __call__(self, training = False):
		"""
		__call__ - execution
		"""

		# parse data
		data = self.parse_train() if training else self.parse_test()

		# batch for batch-processign
		data = data.batch(self.batch_size)

		# preprocessing image and text
		func = self.process_train if training else self.process_test
		data = data.map(func , num_parallel_calls = tf.data.experimental.AUTOTUNE)

		# return data
		return data.prefetch(tf.data.experimental.AUTOTUNE)
