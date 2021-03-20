"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import numpy as np
import tensorflow as tf


"""------------------------------image-processing-------------------------------------------"""
@tf.function
def image_proc(image, size, seed, training):
	"""
	image_proc - function to process image while training. Up to designs
	Inputs:
		- image : Tensor
			List of image tensors
		- size : tuple
			Tuple of (height, width) of the desired image's shape
		- training : boolean
	Outputs:
		- image : Tensor
			Post-processed image
	"""
	# set seed
	np.random.seed(seed)
	
	# transform 
	image = image if not training else image_aug(image, seed)

	# resize to preferred image
	image = tf.image.resize_with_pad(image, size[0], size[1], method = tf.image.ResizeMethod.BILINEAR)

	# standarize
	image = tf.image.per_image_standardization(image)

	return image

def image_aug(img, seed):
	"""
	Randomlly augment image
	Args:
		img : image tensor
		seed : int
	Returns:
		img : image tensor
			augmented images
	"""

	# get random action
	act = np.random.randint(low = 0, high = 3)
	# get shape
	img_shape = tf.shape(img)

	# shift horizontally and vertically within range [-20, 20]
	if act == 0:
		offset_h, offset_w = np.random.randint(low = -20, high = 20), np.random.randint(low = -20, high = 20)
		img = tf.image.crop_to_bounding_box(img,
			0 if offset_h > 0 else abs(offset_h), 0 if offset_w > 0 else abs(offset_w), img_shape[0] - abs(offset_h), img_shape[1] - abs(offset_w))
		img = tf.image.pad_to_bounding_box(img, offset_h if offset_h  > 0 else 0, offset_w if offset_w > 0 else 0, img_shape[0], img_shape[1])
	# ranodmly flip hortizontally
	elif act == 1:
		img = tf.image.random_flip_left_right(img, seed = seed)
	# random brightness
	elif act == 2:
		img = tf.image.random_brightness(img, max_delta = 0.1, seed = seed)

	return img

"""------------------------------text-processing-------------------------------------------"""

@tf.function
def text_proc(text, max_len):
	"""
	text_proc - function to perform fundamental text-processing (not considering WordPiece Tokenizer)
	Inputs:
		- text : Tensor of String
		- max_len : integer
			Maximum number of tokens in a sequence
	Outputs:
		- text :  Tensor of string
			Post-processed string
	"""

	# lower case
	text = tf.strings.lower(text)

	# flatten punctuations and short-forms
	text = regex_replace(text)

	# remove trivial whitespace
	text = tf.strings.regex_replace(text, pattern = "\s+", rewrite = " ")

	# join string together
	text = tf.strings.reduce_join(text, separator = '[SEP]')

	# tokenize (split by space)
	text = tf.strings.split(text)

	# padding 
	text = tf.cond(pred = tf.math.greater(tf.size(text), max_len),
		true_fn = lambda: tf.slice(text, begin = [0], size = [max_len]),
		false_fn = lambda : pad_text(text, max_len))

	return text

@tf.function
def text_blank(text, text_len):
	"""
	Randomly blank tokens
	Args;
		text : tensor
		text_len : int
	Returns:
		text : tensor
	"""

	# randomly generate blank indices
	blank_index = np.random.randint(low = 1, high = text_len, size = np.random.randint(low = 0, high 4))
	blank_index = tf.one_hot(indices = blank_index, depth = 1, on_value = 0, off_value = 1, dtype = tf.int32)

	# blank tokens
	text = tf.math.multiply(text, blank_index)

	return text

@tf.function
def pad_text(text, max_len):
	"""
	pad_text - function to pad [PAD] token to string
	Inputs:
		- text : Tensor of string
		- max_len : integer
			Maximum number of tokens in a sequence
	Outputs: 
		- _ : Tensor of string
			Post-processed string
	"""
	paddings = tf.repeat(tf.constant('[PAD]'),
		repeats = max_len - tf.size(text))

	return tf.concat([text, paddings], axis = 0)

@tf.function
def regex_replace(text):
	"""
	regex_replace - function to flatten punctuations and short-forms
	Inputs:
		- text : Tensor of string
	Outputs:
		- _ : Tensor of String
			Post-processed string 
	"""
	def _func(inputs):
		"""
		_func - function to perform regex-replace
		"""
		# replace n't with not
		inputs = tf.strings.regex_replace(inputs,
			pattern = "n't",
			rewrite = " not")

		# replace: 'm, 's, 're with be
		inputs = tf.strings.regex_replace(inputs,
			pattern = "\'s|\'re|\'m",
			rewrite = " be")

		# replace punctuations with [PUNC] mark
		inputs = tf.strings.regex_replace(inputs,
			pattern = "[^a-zA-Z\d\s]",
			rewrite = " [PUNC] ")

		return inputs

	return _func(text) if text.dtype.is_compatible_with(tf.string) else tf.constant("")
