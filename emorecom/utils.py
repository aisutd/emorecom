"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

"""------------------------------image-processing-------------------------------------------"""
@tf.function
def image_proc(image, size, overlap_ratio):
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

	# resize iamge
	#image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])

	# split image to chunks
	image = image_to_chunks(image, size[0], size[1], overlap_ratio)

	# resize image
	image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])

	# standarize
	image = tf.image.per_image_standardization(image)

	return image

def image_to_chunks(image, height, width, overlap_ratio):
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

	"""----cut-images-into-smaller-chunks-vertically-----"""
	img_shape = tf.shape(image)
	img_h, img_w = img_shape[0], img_shape[1]

	def _crop():
		"""
		_crop - function to split images into smaller chunks.
			If height > ratio x width, then split by height. And vice versa
			Default, split by horizontally
		"""
		if img_h > 2 * img_w:
			# split by height
			num_split = tf.cast(tf.math.ceil(img_h / height), dtype = tf.int32)
			delta = tf.cast(tf.math.round(img_h / num_split), dtype = tf.int32)
			outputs = tf.zeros([1, height, width, 3])
			cond = lambda idx, o: idx < num_split
			body = lambda idx, o: [idx + 1, tf.concat([o,
					tf.expand_dims(crop_and_pad(image, img_h, img_w, idx * delta, (idx + 1) * delta, 0, width, height, width), axis = 0)], axis = 0)]
			_, outputs = tf.while_loop(cond = cond, body = body,
				loop_vars = [0, outputs], shape_invariants = [None, tf.TensorShape([None, height, width, 3])])
			return outputs #[1:]
		else:
			# split by width
			num_split = tf.cast(tf.math.ceil(img_w / width), dtype = tf.int32)
			delta = tf.cast(tf.math.round(img_w / num_split), dtype = tf.int32)

			outputs = tf.zeros([1, height, width, 3])
			cond = lambda idx, o: idx < num_split
			body = lambda idx, o: [idx + 1, tf.concat([o,
					tf.expand_dims(crop_and_pad(image, img_h, img_w, 0, height, idx * delta, (idx + 1) * delta, height, width), axis = 0)], axis = 0)]
			_, outputs = tf.while_loop(cond = cond, body = body,
				loop_vars = [0, outputs], shape_invariants = [None, tf.TensorShape([None, height, width, 3])])

			return outputs#[1:]

	img = tf.cond(pred = tf.math.logical_and(tf.math.less(img_h, height), tf.math.less(img_w, width)),
		true_fn = lambda: tf.image.resize_with_pad(image, height, width),
		false_fn = lambda: _crop())

	tf.print(tf.shape(img), tf.shape(image))
	return image

@tf.function
def crop_and_pad(image, img_h, img_w, x1, x2, y1, y2, height, width):
	"""
	crop_and_pad - function top crop image to the desired size and pad them if necessary
	"""

	# crop
	image = image[x1:tf.math.minimum(x2, img_h),
		y1:tf.math.minimum(y2, img_w), :]

	# resize 
	image = tf.image.resize_with_crop_or_pad(image, height, width)
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
