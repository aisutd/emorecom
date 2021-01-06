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

	def __init__(self, data, vocabs, text_len, image_size, overlap_ratio = 0.3,  batch_size = 1, buffer_size = 512, seed = 2021):
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
			- overlap_ratio : float
				To be added
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
		Outputs:
			- _ : Tensor Look-up table
		"""

		initializer =  tf.lookup.TextFileInitializer(filename = file,
			key_dtype = tf.string, key_index = tf.lookup.TextFileIndex.WHOLE_LINE,
			value_dtype = tf.int64, value_index = tf.lookup.TextFileIndex.LINE_NUMBER)

		return tf.lookup.StaticHashTable(initializer, default_value = 0)

	def parse_train(self):
		"""
		parse_train - function to parse training data in TFRecord format
		Inputs: None
		Outputs:
			- data : Tensor Dataset
		"""
		# shuffle data
		data = self.data.shuffle(buffer_size = self.buffer_size, reshuffle_each_iteration = True)

		# read data
		@tf.function
		def _parse(example):
			example = tf.io.parse_single_example(example, self.train_features)

			# read image
			example['image'] = tf.io.read_file(example['image'])
		
			return {'image' : example['image'], 'transcripts' : example['transcripts']}, example['label']
		data = data.cache().map(_parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)

		return data

	def parse_test(self):
		"""
		parse_test - function to parse testing data in TFRecord format
		Inputs: None
		Outputs:
			- data : Tensor Dataset
		"""

		# read data
		@tf.function
		def _parse(example):
			example = tf.io.parse_single_example(example, self.test_features)

			# read image
			example['image'] = tf.io.read_file(example['image'])

			return {'image' : example['image'], 'transcripts' : example['transcripts']}
		data = self.data.cache().map(_parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)
		return data

	@tf.function
	def _image(self, image):
		"""
		process image
		Inputs:
			- image : Tensor
				Image 
		Outputs:
			- image : Tensor
				Post-processed image
		"""

		# decode image
		image = tf.io.decode_image(image, dtype = tf.float32)

		# process image
		#image = tf.map_fn(fn = lambda img : image_proc(img, size = self.image_size, overlap_ratio = self.overlap_ratio),
		#	elems = image, fn_output_signature = tf.float32)

		# below is for non-batching
		image = image_proc(image, size = self.image_size, overlap_ratio = self.overlap_ratio)

		return image

	@tf.function
	def _transcripts(self, transcript):
		"""
		process transcripts
		Inputs:
			- transcript : Tensor of string
				List of transcripts separated by ;
		Outputs:
			- transcript : Tensor of string
				Post-processed transcript
		"""

		# split transcripts
		transcript = tf.strings.split(transcript, sep = ';')

		# processing: lowercase, strip whitepsaces, tokenize, and padding
		#transcript = tf.map_fn(fn = lambda x: text_proc(x, self.text_len), elems = transcript,
		#	fn_output_signature = tf.string)
		transcript = text_proc(transcript, self.text_len)

		# decode vocab-index
		transcript = self.vocabs.lookup(transcript)

		return transcript

	@tf.function
	def _label(self, label):
		"""
		process label
		Inputs:
			- label : Tensor
				String: list of one-hot encoded labels for 8 emotion types
		Outputs:
			- label : Tensor
				Post-processed label (one-hot encoded)
		"""

		# split lable by ','
		label = tf.strings.split(label, sep = ',')

		# convert str to integer
		#print(label)
		#tf.print(label)
		label = tf.strings.to_number(label, out_type = tf.int32)#.to_tensor()

		return label

	@tf.function
	def process_train(self, features, labels):
	
		"""
		process_train - function to preprocess image, text, and label
		Inputs:
			- features : dictionary of Tensor
				Dict of {'image' : Tensor, 'transcripts' : Tensor}
			- labels : Tensor
				Tensor of labels
		Outputs:
			- _ : dictionary of Tensor
				Dict of {'image' : Tensor, 'transcripts' : Tensor}
			- _ : Tensor
				Tensor of labels
		"""
		return {'image' : self._image(features['image']), 'transcripts' : self._transcripts(features['transcripts'])}, self._label(labels)

	@tf.function
	def process_test(self, features):
		"""
		process_test - function to preprocess image and text only
		Inputs: 
			- features : dictionary of Tensor
				Dict of {'image' : Tensor, 'transcripts' : Tensor}
		Outputs:
			- _ : dictionary of Tensor
				Dict of {'image' : Tensor, 'transcripts' : Tensor}
		"""

		return {'image' : self._image(features['image']), 'transcripts' : self._transcripts(features['transcripts'])}

	def __call__(self, training = False):
		"""
		__call__ - function to create a Dataset instnace
		Inputs:
			- training : boolean
				Boolean value to return a training or testing Dataset instnace
		Outputs: None
		"""

		# parse data
		data = self.parse_train() if training else self.parse_test()

		# batch
		#data = data.batch(self.batch_size, drop_remainder = True)

		# preprocessing image and text
		if training:
			#data = data.map(lambda features, labels: self.process_train(features, labels),
			data = data.map(self.process_train,
				num_parallel_calls = tf.data.experimental.AUTOTUNE)
			data = data.batch(self.batch_size, drop_remainder = True)
		else:
			#data = data.map(lambda features: self.process_test(features),
			data = data.map(self.process_test,
				num_parallel_calls = tf.data.experimental.AUTOTUNE)
			data = data.batch(self.batch_size, drop_remainder = True)

		return data.prefetch(tf.data.experimental.AUTOTUNE)
