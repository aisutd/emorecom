"""
utils.py - data-preprocessing module
"""

# import dependencies
import re
import string
import tensorflow as tf

@tf.function
def basic_text_proc(input):

	# lower case
	input = tf.strings.lower(input)

	# flatten punctuations and short-forms
	input = regex_replace(input)

	# tokenize (split by space)
	input = tf.strings.split(input)
	
	return input

@tf.function
def regex_replace(input):
	"""
	regex_replace - function to flatten punctuations and short-forms
	"""

	# replace n't with not
	input = tf.strings.regex_replace(input,
		pattern = "n't",
		rewrite = " not")

	# replace: 'm, 's, 're with be
	input = tf.strings.regex_replace(input,
		pattern = "'s|'re|'m",
		rewrite = " be")

	tf.print(input)

	# replace punctuations with [PUNC] mark
	input = tf.strings.regex_replace(input,
		pattern = "[^a-zA-Z\d\s]",
		rewrite = " [PUNC]")

	return input
