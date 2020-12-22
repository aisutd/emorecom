"""
model.py - module for developing model
"""

# import dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding

# set random seed
tf.random.set_seed(2021)
np.random.seed(seed = 2021)

def create_model(configs):

	# vision
	vision_model = vision(img_shape = configs['img_shape'])

	# text
	text_model = text(text_shape = configs['text_shape'],
		vocab_size = configs['vocab_size'], vocabs = configs['vocabs'],
		max_len = configs['max_len'], embed_dim = configs['embed_dim'],
		pretrained_embed = configs['pretrained_embed'])

	return Model(inputs = [vision_model.inputs, text_model.inputs],
		outputs = [vision_model.outputs, text_model.outputs])

def vision(img_shape):
	"""
	vision - function to create visual module
	Inputs:
		- img_shape : tuple of integers
			[width, height, channel]
	"""
	inputs = Input(shape = img_shape)

	outputs = tf.keras.applications.ResNet50(include_top = False,
		weights = 'imagenet', input_shape = img_shape)(inputs)

	return Model(inputs = inputs, outputs = outputs)

def BiLSTM(forward_units, backward_units):
	forward = LSTM(forward_units, return_sequences = True)
	backward = LSTM(backward_units, return_sequences = True, go_backwards = True)

	return Bidirectional(layer = forward, backward_layer = backward, merge_mode = 'concat')

def EmbeddingLayer(vocab_size, embed_dim, vocabs, max_len = None, pretrained = None):

	if pretrained:
		# retrieve pretrained word embeddings
		embed_index = {}
		with open(pretrained) as file:
			for line in file:
				word, coefs = line.split(maxsplit = 1)
				coefs = np.fromstring(coefs, 'f', sep = ' ')
				embed_index[word] = coefs
		# initialize new word embedding matrics
		embed_dim = len(list(embed_index.values())[0])
		embeds = np.random.uniform(size = (vocab_size, embed_dim))
		
		# parse words to pretrained word embeddings
		words, indices = vocabs
		for text, i in zip(words.numpy(), indices.numpy()):
			word = str(text)[2:-1] # get word from byte-class string
			embed = embed_index.get(word)
			if embed:
				embeds[i] = embed

		initialilzer = Constant(embeds)
	else:
		initializer = 'uniform'

	return Embedding(input_dim = vocab_size + 1, output_dim = embed_dim,
		input_length = max_len, mask_zero = True,
		embeddings_initializer = initializer,
		embeddings_regularizer = None)
	
def text(text_shape, vocab_size, vocabs, max_len = None, embed_dim = None, pretrained_embed = None):
	"""
	text - function to create textual module
	Inputs:
		- text_shape : tuple of integers
			(max_seq_length, embedding_size)
		- vocab_size : integer
			Number of vocabs
		- vocabs : dict
			Dictionary of words-index
		- max_len : integer, None by defualt
			Maximum length of the input
		- pretrained_embed : str, None by default
			Path to the pretrained embedding file
	"""

	# initialize input
	inputs = Input(shape = text_shape)

	# initializer Embedding layer
	embeddings = EmbeddingLayer(vocab_size = vocab_size, embed_dim = embed_dim,
		vocabs = vocabs, max_len = max_len, pretrained = pretrained_embed)(inputs)

	tf.print(embeddings.shape)
	# bidirectional-lstm
	outputs = BiLSTM(128, 128)(embeddings)

	return Model(inputs = inputs, outputs = outputs)
