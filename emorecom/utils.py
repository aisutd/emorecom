"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

@tf.function
def basic_text_proc(input, max_len):
	"""
	basic_text_proc - function to perform fundamental text-processing (not considering WordPiece Tokenizer)
	"""

	# lower case
	input = tf.strings.lower(input)

	# strip whitespaces
	input = tf.strings.strip(input)

	# flatten punctuations and short-forms
	input = regex_replace(input)

	# join string together
	input = tf.strings.reduce_join(input, separator = ' ')

	# tokenize (split by space)
	#tf.print('basic-text-proc', input, tf.size(input))
	input = tf.strings.split(input)

	# padding 
	input = tf.cond(pred = tf.math.greater(tf.size(input), max_len),
		true_fn = lambda: tf.slice(input, begin = [0], size = [max_len]),
		false_fn = lambda : pad_text(input, max_len))

	# check final result
	#tf.print('split', input, tf.shape(input), tf.size(input))

	return input

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
	def _func(input):
		"""
		_func - function to perform regex-replace
		"""
		# replace n't with not
		input = tf.strings.regex_replace(input,
			pattern = "n't",
			rewrite = " not")

		# replace: 'm, 's, 're with be
		input = tf.strings.regex_replace(input,
			pattern = "'s|'re|'m",
			rewrite = " be")

		# replace punctuations with [PUNC] mark
		input = tf.strings.regex_replace(input,
			pattern = "[^a-zA-Z\d\s]",
			rewrite = " [PUNC]")

		return input
	return _func(text) if text.dtype.is_compatible_with(tf.string) else tf.constant("")
