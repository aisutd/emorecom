"""
model.py - module for developing model
"""

# import dependencies
import tensorflow as tf

from tensorflow.keras import Input, Model

def create_model(configs):
	return Model(inputs = inputs, outputs = outputs)

def vision():
	"""
	vision - function to create visual module
	"""
	return Model(inputs, outputs = outputs)

def text():
	"""
	text - function to create textual module
	"""
	return Model(inputs = inputs, outputs = outputs)
