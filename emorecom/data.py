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

	# initialize global feature dictionary
	train_features = {
		'image' : tf.io.FixedLenFeature([], tf.string),
		'transcripts' : tf.io.FixedLenFeature([], tf.string),
		'label' : tf.io.FixedLenFeature([], tf.string)}

	test_features {
		'image' : tf.io.FixedLenFeature([], tf.string),
		'transcripts' : tf.io.FixedLenFeature([], tf.string)}

	def __init__(self, data_path, batch_size = 1):
		"""
		Class constructor:
		Inputs:
			- data_path : str or list of str
				Path(s) to TFRecord dataset
		"""

		self.data = tf.data.TFRecordDataset(data_path)
		self.batch_size = batch_size

	def parse_train():
		"""
		parse_train - function to parse training data in TFRecord format
		"""
		return None

	def parse_test():
		"""
		parse_test - function to parse testing data in TFRecord format
		"""
		return None

	def __call__(self, training = False):
		"""
		__call__ - execution
		"""
		# parse data
		data = self.parse_train() if training else self.parse_test()
