"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

@tf.function
def basic_text_proc(inputs, max_len):
	"""
	basic_text_proc - function to perform fundamental text-processing (not considering WordPiece Tokenizer)
	"""

	# lower case
	inputs = tf.strings.lower(inputs)

	# flatten punctuations and short-forms
	inputs = regex_replace(inputs)

	# remove trivial whitespace
	inputs = tf.strings.regex_replace(inputs, pattern = "\s+", rewrite = " ")

	# join string together
	inputs = tf.strings.reduce_join(inputs, separator = '[SEP]')

	# tokenize (split by space)
	#tf.print('basic-text-proc', input, tf.size(input))
	inputs = tf.strings.split(inputs)

	# padding 
	inputs = tf.cond(pred = tf.math.greater(tf.size(inputs), max_len),
		true_fn = lambda: tf.slice(inputs, begin = [0], size = [max_len]),
		false_fn = lambda : pad_text(inputs, max_len))

	# check final result
	#tf.print('split', inputs, tf.shape(inputs), tf.size(inputs))

	return inputs

@tf.function
def pad_text(input, max_len):
	"""
	pad_text - function to pad [PAD] token to string
	"""
	paddings = tf.repeat(tf.constant('[PAD]'),
		repeats = max_len - tf.size(input))

	return tf.concat([input, paddings], axis = 0)
	


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
