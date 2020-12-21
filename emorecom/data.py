"""
data.py - data-loading module
"""

# import dependencies
import os
import tensorflow as tf
import cv2

# import local packages


class Dataset:
	"""
	Dataset - class to implement Tensorflow Data API
	"""

	def __init__(self, data_path, batch_size = 1):
		"""
		Class constructor:
		Inputs:
			- data_path : str or list of str
				Path(s) to TFRecord dataset
		"""

		# parse arguments
		self.data = tf.data.TFRecordDataset(data_path)
		self.batch_size = batch_size

		# initialize global feature dictionary
		self.train_features = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string),
			'label' : tf.io.FixedLenFeature([], tf.string)}

		self.test_features = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string)}

	def parse_train(self):
		"""
		parse_train - function to parse training data in TFRecord format
		"""

		# read data
		@tf.function
		def _parse(example):
			example = tf.io.parse_single_example(example, self.train_features)
			return {'image' : example['image'], 'transcripts' : example['transcripts'], 'label' : example['label']}
		data = self.data.cache().map(_parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)

		# batch
		data = data.batch(self.batch_size)

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

		# batch
		data = data.batch(self.batch_size)
		
		return data

	def __call__(self, training = False):
		"""
		__call__ - execution
		"""

		# parse data
		data = self.parse_train() if training else self.parse_test()

		# preprocessing image and text
		return data.prefetch(tf.data.experimental.AUTOTUNE)
