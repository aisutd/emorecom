"""
model.py - module for developing model
"""

# import dependencies
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional

def create_model(configs):

	# vision
	vision_model = vision(img_shape = configs['img_shape'])

	# text
	text_model = text(text_shape = configs['text_shape'])

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

	
def text(text_shape):
	"""
	text - function to create textual module
	Inputs:
		- text_shape : tuple of integers
			(max_seq_length, embedding_size)
	"""

	inputs = Input(shape = text_shape)

	# bidirectional-lstm
	outputs = BiLSTM(128, 128)(inputs)

	return Model(inputs = inputs, outputs = outputs)
