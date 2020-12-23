"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

@tf.function
def basic_text_proc(input):
	"""
	basic_text_proc - function to perform fundamental text-processing (not considering WordPiece Tokenizer)
	"""

	# lower case
	input = tf.strings.lower(input)

	# flatten punctuations and short-forms
	input = regex_replace(input)

	# tokenize (split by space)
	input = tf.strings.split(input)
	
	return input

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
