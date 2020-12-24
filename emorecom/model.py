"""
model.py - module for developing model
"""

# import dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Dense, Flatten, LSTM, Bidirectional, Embedding

# set random seed
tf.random.set_seed(2021)
np.random.seed(seed = 2021)

def create_model(configs):

	# vision
	vision_model = vision(img_shape = configs['img_shape'])

	# text
	text_model = text(text_shape = configs['text_shape'],
		vocabs = configs['vocabs'], max_len = configs['max_len'],
		vocab_size = configs['vocab_size'], embed_dim = configs['embed_dim'],
		pretrained_embed = configs['pretrained_embed'])

	# fuse visiual and textual features
	vision_features = Conv1D(256, kernel_size = 3, strides = 1, activation = 'relu')(vision_model.outputs[0])
	shape = tf.shape(vision_features)
	vision_features = tf.reshape(vision_features, shape = [shape[0], -1, shape[-1]])

	outputs = tf.concat([vision_features, text_model.outputs[0]], axis = 1,
		name = 'fusion-concat')

	# classfication module
	outputs = Dense(128, activation = 'relu')(outputs)
	outputs = Dropout(0.2)(outputs)
	outputs = Dense(64, activation = 'relu')(outputs)

	outputs = Dense(configs['num_class'])(outputs)
	return Model(inputs = [vision_model.inputs, text_model.inputs],
		outputs = outputs)

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

def EmbeddingLayer(embed_dim = None, vocabs = None, vocab_size = None, max_len = None, pretrained = None):

	# retrieve vocab-size
	# index-0 for out-of-vocab token
	if vocabs:
		vocab_size = len(vocabs) + 1
	else:
		assert vocab_size != None
		vocab_size += 1

	# load pretrained word-embeddings
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
		for text, i in zip(vocabs, range(1, vocab_size + 1)):
			embed = embed_index.get(word)
			if embed is not None:
				embeds[i] = embed

		initializer = Constant(embeds)
	else:
		initializer = 'uniform'

	return Embedding(input_dim = vocab_size, output_dim = embed_dim,
		input_length = max_len, mask_zero = True,
		embeddings_initializer = initializer,
		embeddings_regularizer = None)
	
def text(text_shape, vocabs, vocab_size = None, max_len = None, embed_dim = None, pretrained_embed = None):
	"""
	text - function to create textual module
	Inputs:
		- text_shape : tuple of integers
			(max_seq_length, embedding_size)
		- vocabs : str
			Path to dictionary file
		- vocab_size : integer
			Number of vocabs, None by default
		- max_len : integer, None by defualt
			Maximum length of the input
		- pretrained_embed : str, None by default
			Path to the pretrained embedding file
	"""

	# read vocabs
	with open(vocabs) as file:
		vocabs = file.read().split('\n')[:-1]

	# initialize input
	inputs = Input(shape = text_shape)

	# initializer Embedding layer
	embeddings = EmbeddingLayer(embed_dim = embed_dim, vocabs = vocabs,
		vocab_size = vocab_size, max_len = max_len, pretrained = pretrained_embed)(inputs)

	# bidirectional-lstm
	outputs = BiLSTM(128, 128)(embeddings)

	return Model(inputs = inputs, outputs = outputs)
