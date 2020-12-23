"""
utils.py - data-preprocessing module
"""

# import dependencies
import nltk
import tensorflow as tf

def tokenize(input):

	# check input as tensor
	assert isinstance(input, tf.Tensor)

	# lower
	input = tf.strings.lower(input)
	
	# replace punctuations with space + punc
	input = tf.strigns.regex_replace(input, pattern = , rewrite = rewrite, replace_global = True)
