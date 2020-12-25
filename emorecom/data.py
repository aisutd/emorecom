"""
data.py - data-loading module
"""

# import dependencies
import os
import cv2
import pickle
import tensorflow as tf

# import local packages
from emorecom.utils import text_proc, image_proc

class Dataset:
	"""
	Dataset - class to implement Tensorflow Data API
	"""

	def __init__(self, data, vocabs, text_len, image_size, overlap_ratio = 0.3, batch_size = 1, buffer_size = 512, seed = 2021):
		"""
		Class constructor:
		Inputs:
			- data : str or list of str
				Path(s) to TFRecord dataset
			- vocabs : str
				Path to vocabs dictionary
			- text_len : integer
				Maximum length of text
			- image_size : tuple of integers
				Tuple of [height, width]
			- overlap_ratio float
				Ratio of overlapping. If smaller, then ignore the cropped image
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
		self.data = tf.data.TFRecordDataset(data) # cache data
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.text_len = text_len
		self.image_size = image_size
		self.overlap_ratio = overlap_ratio

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

		initializer =  tf.lookup.TextFileInitializer(filename = file,
			key_dtype = tf.string, key_index = tf.lookup.TextFileIndex.WHOLE_LINE,
			value_dtype = tf.int64, value_index = tf.lookup.TextFileIndex.LINE_NUMBER)

		return tf.lookup.StaticHashTable(initializer, default_value = 0)

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

			return example['image'], example['transcripts'], example['label']
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
			return example['image'], example['transcripts']
		data = data.cache().map(_parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)
		return data

	@tf.function
	def _image(self, image):
		"""
		process image
		"""

		# process image
		image = tf.map_fn(fn = lambda img : image_proc(img, size = self.image_size, overlap_ratio = self.overlap_ratio),
			elems = image, fn_output_signature = tf.float32)

		# below is for non-batching
		#imgae = image_proc(image, size = self.image_size, overlap_ratio = self.overlap_ratio)

		return image

	@tf.function
	def _transcripts(self, input):
		"""
		process transcripts
		"""

		# split transcripts
		input = tf.strings.split(input, sep = ';')

		# check split transcripts
		#tf.print('text', input, tf.size(input), input.shape)

		# processing: lowercase, strip whitepsaces, tokenize, and padding
		input = tf.map_fn(fn = lambda x: text_proc(x, self.text_len), elems = input,
			fn_output_signature = tf.string)
		#tf.print('tokenized', input, tf.size(input), tf.shape(input))

		# decode vocab-index
		input = self.vocabs.lookup(input)
		#tf.print("decoded", input, tf.size(input), tf.shape(input))

		return input

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

	#@tf.function
	def process_train(self, image, transcripts, labels):
		"""
		process_train - function to preprocess image, text, and label
		"""
		return self._image(image), self._transcripts(transcripts), self._label(labels)

	@tf.function
	def process_test(self, image, transcripts):
		"""
		process_test - function to preprocess image and text only
		"""

		return self._image(image), self._transcripts(transripts)

	def __call__(self, training = False):
		"""
		__call__ - execution
		"""

		# parse data
		data = self.parse_train() if training else self.parse_test()

		# batch
		data = data.batch(self.batch_size)

		# preprocessing image and text
		func = self.process_train if training else self.process_test
		data = data.map(lambda image, transcripts, labels: func(image, transcripts, labels), num_parallel_calls = tf.data.experimental.AUTOTUNE)
		"""
		if training:
			data = data.map(lambda image, transcripts, labels: [self._image(image), self._transcripts(transcripts), self._label(labels)],
				num_parallel_calls = tf.data.experimental.AUTOTUNE)
		else:
			data = data.map(lambda image, transcripts: [self._image(image), self._transcripts(transcripts)],
				num_parallel_calls = tf.data.experimental.AUTOTUNE)
		"""

		# return data
		return data#.prefetch(tf.data.experimental.AUTOTUNE)
