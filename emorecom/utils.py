"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

"""------------------------------image-processing-------------------------------------------"""
@tf.function
def image_proc(image, size):
	"""
	image_proc - function to process image while training. Up to designs
	Inputs:
		- image : Tensor
			List of image tensors
		- size : tuple
			Tuple of (height, width) of the desired image's shape
	"""
	
	# load image
	image = tf.io.decode_image(image, dtype = tf.float32)

	# resize
	image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])

	# split image to chunks
	#image = image_to_chunks(image, size[0], size[1])

	# standarize
	image = tf.image.per_image_standardization(image)

	return image

def image_to_chunks(image, height, width):
	"""
	image_to_chunks - split a large image to smaller images
	Inputs:
		- image : Tensor
			Tensor of image in shape [width, height, 3]
		- height : integer
			Target height
		- width : integer
			Target width
	"""

	"""----cut-images-into-smaller-chunks"""
	img_h, img_w, _ = tf.shape(image)
	h_indices = tf.range(0, img_h, size[0], dtype = tf.int32)
	w_indices = tf.range(0, img_w, size[1], dtype = tf.int32)
	box_indices = tf.stack([h_indices, w_indices], axis = -1)

	return image

def crop_and_pad(image, img_h, img_w, h_offset, w_offset, height, width):
	"""
	crop_and_pad - function top crop image to the desired size and pad them if necessary
	"""

	#image = image[h_offset:tf.math.maximum(h_offset + height, img_h), w_offset:tf.math.maxmumw_offset + width, img_w), :]

	return image

"""------------------------------text-processing-------------------------------------------"""
@tf.function
def text_proc(text, max_len):
	"""
	text_proc - function to perform fundamental text-processing (not considering WordPiece Tokenizer)
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
	#tf.print('basic-text-proc', input, tf.size(input))
	text = tf.strings.split(text)

	# padding 
	text = tf.cond(pred = tf.math.greater(tf.size(text), max_len),
		true_fn = lambda: tf.slice(text, begin = [0], size = [max_len]),
		false_fn = lambda : pad_text(text, max_len))

	# check final result
	#tf.print('split', inputs, tf.shape(inputs), tf.size(inputs))

	return text

@tf.function
def pad_text(text, max_len):
	"""
	pad_text - function to pad [PAD] token to string
	"""
	paddings = tf.repeat(tf.constant('[PAD]'),
		repeats = max_len - tf.size(text))

	return tf.concat([text, paddings], axis = 0)

@tf.function
def regex_replace(text):
	"""
	regex_replace - function to flatten punctuations and short-forms
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
