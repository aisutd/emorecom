"""
model.py - module for developing model
"""

# import dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Reshape, GlobalAveragePooling2D, GlobalAveragePooling1D, Conv2D, Dropout, BatchNormalization, Dense, Flatten, LSTM, Bidirectional, Embedding

# set random seed
tf.random.set_seed(2021)
np.random.seed(seed = 2021)

def create_model(configs):
	"""
	create_model - function to create Visual-Textual model for Emotion Recognitin on Comic Scenes
	Inputs:
		- configs : dict
			Dicitonary of model configurations
	Outputs:
		- _ : Tensorflow Keras Model
	"""

	# vision
	vision_model = vision(img_shape = configs['img_shape'])

	# text
	text_model = text(text_len = configs['text_len'], vocabs = configs['vocabs'],
		vocab_size = configs['vocab_size'], embed_dim = configs['embed_dim'],
		pretrained_embed = configs['pretrained_embed'])

	# fuse visiual and textual features
	vision_features = Conv2D(512, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(vision_model.outputs[0])
	vision_features = Reshape((-1, 512))(vision_features)

	#tf.print("vision-shape {} and text-shape {}".format(vision_features.shape, text_model.outputs[0].shape))
	outputs = tf.concat([vision_features, text_model.outputs[0]], axis = 1,
		name = 'fusion-concat')	

	# select max-features
	#outputs = tf.keras.layers.AveragePooling1D()(text_model.outputs[0])
	#outputs = text_model.outputs[0]

	# classfication module
	outputs = Dense(256, activation = 'relu', kernel_regularizer = 'l2')(outputs)
	outputs = Flatten()(outputs)
	outputs = Dense(128, activation = 'relu', kernel_regularizer = 'l2')(outputs)
	#outputs = Dense(64, activation = 'relu', kernel_regularizer = 'l2')(outputs)
	outputs = Dense(configs['num_class'], activation = 'sigmoid')(outputs)

	return Model(inputs = [vision_model.inputs, text_model.inputs],
		outputs = outputs)

def vision(img_shape):
	"""
	vision - function to create visual module
	Inputs:
		- img_shape : tuple of integers
			[width, height, channel]
	Outputs:
		- _ : Tensorflow Keras Model
	"""
	inputs = Input(shape = img_shape, name = 'image')

	outputs = tf.keras.applications.ResNet50(include_top = False,
		weights = 'imagenet', input_shape = img_shape)(inputs)

	return Model(inputs = inputs, outputs = outputs)

def BiLSTM(forward_units, backward_units, return_sequences = True):
	"""
	BiLSTM - function to create the Bidirection-LSTM layer
	Inputs:
		- forward_units : integer
			Number of units for the forward direction
		- bachward_units : integer
			Number of units for the baackward direction
	Outputs:
		- _ : Tensorflow Keras Layer - Bidirectional object
	"""
	forward = LSTM(forward_units, return_sequences = return_sequences)
	backward = LSTM(backward_units, return_sequences = return_sequences, go_backwards = True)

	return Bidirectional(layer = forward, backward_layer = backward, merge_mode = 'concat')

def EmbeddingLayer(embed_dim = None, vocabs = None, vocab_size = None, max_len = None, pretrained = None):
	"""
	EmbeddingLayer - function to initializer Embedding layer
	Inputs:
		- embed_dim : integer
			Dimension size of the Embedding layer
		- vocabs : path to vocab file
		- vocab_size : number of vocabularies
			If vocabs is None, vocabo_size must be a valid integer
		- max_len : integer
			Maximum number of tokens in a sequence
		- pretrained : string
			Path to the pretraiend word embeddings. Glove Word-Embedding is used by default
	Outputs:
		- _ : Tensorflow Embedding layer
	"""

	# retrieve vocab-size
	# index-0 for out-of-vocab token
	if vocabs:
		# read vocabs
		with open(vocabs) as file:
			vocabs = file.read().split('\n')[:-1]
		vocab_size = len(vocabs)
	else:
		# if vocabs not given, then vocab-size must not be None
		assert vocab_size != None

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
		for i, word in enumerate(vocabs):
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
	
def text(text_len = None, vocabs = None, vocab_size = None, embed_dim = None, pretrained_embed = None):
	"""
	text - function to create textual module
	Inputs:
		- text_len : integer
			Max length of text, default = None
		- vocabs : str
			Path to dictionary file
		- vocab_size : integer
			Number of vocabs, None by default
		- embed_dim : integer
			Dimension size of the Embedding layer
		- pretrained_embed : str, None by default
			Path to the pretrained embedding file
	Outputs:
		- _ : Tensorflow Keras Model
	"""

	# initialize input
	inputs = Input(shape = [text_len], name = 'transcripts')

	# initializer Embedding layer
	embeddings = EmbeddingLayer(embed_dim = embed_dim, vocabs = vocabs,
		vocab_size = vocab_size, max_len = text_len, pretrained = pretrained_embed)(inputs)

	# bidirectional-lstm
	outputs = BiLSTM(256, 256)(embeddings)
	#outputs = BiLSTM(256, 256)(outputs)

	return Model(inputs = inputs, outputs = outputs)
