"""
predict.py - prediction module
""" 
# import dependencies
import os
import glob
import argparse
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks, losses, metrics

# import local packages
from emorecom.data import Dataset

# get directory path
DIR_PATH = os.getcwd()

# reset session
tf.keras.backend.clear_session()

def _load_model(path):
	"""
	_load_model - function load model for prediction
	Inputs:
		- path : str
			Path to to-be-loaded model
	Ouptuts:
		- model : tf.keras.Model
	"""

	# load model
	model = tf.keras.models.load_model(path)

	# freeeze model
	for idx in range(len(model.layers)):
		model.layers[idx].trainable = False
		model.layers[idx].training = False

	return model

def main(args):

	"""------parser-arugments------"""
	# initialize dataset
	assert args.test_data
	TEST_DATA = os.path.join(DIR_PATH, args.test_data)
	
	# initialize train-dataset
	print("Creating Data Loading")
	VOCABS = os.path.join(DIR_PATH, args.vocabs)
	dataset = Dataset(data = TEST_DATA, vocabs = VOCABS, image_size = [args.image_height, args.image_width],
		text_len = args.text_len, batch_size = args.batch_size)
	test_data = dataset(training = False)

	# inspect test-dataset
	iterator = iter(test_data)
	features = next(iterator)
	for _ in range(8):
		print(features['image'].shape, features['transcripts'].shape)
		try:
			feautres = next(iterator)
		except:
			break

	# initialize model
	"""
	print("Initialize and compile model")
	MODEL_CONFIGS= {'img_shape' : [args.image_height, args.image_width, 3],
		'text_len' : args.text_len,
		'vocabs' : VOCABS,
		'vocab_size' : args.vocab_size,
		'embed_dim' : args.embedding_dim,
		'pretrained_embed' : os.path.join(DIR_PATH, args.pretrained_embedding),
		'num_class' : args.num_class}
	model = create_model(configs = MODEL_CONFIGS)
	"""
	model_path = os.path.join(DIR_PATH, args.saved_models, args.experiment_name)
	model = _load_model(model_path)
	print(model.summary())

	# set hyperparameters
	"""
	OPTIMIZER = optimizers.Adam(learning_rate = args.learning_rate)
	LOSS = losses.BinaryCrossentropy(from_logits = False)
	METRICS = [metrics.BinaryAccuracy(),
		metrics.Precision(),
		metrics.Recall(),
		metrics.AUC(multi_label = True, thresholds = [0.5])]

	# compile model
	model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)
	"""

	# make predictions
	predictions = model.predict(test_data, verbose = 1)	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	# add arguments
	parser.add_argument('--experiment-name', type = str, help = 'Name of the modle')
	parser.add_argument('--saved-models', type = str, default = 'saved_models', help = 'Path to saved_model for prediction')
	parser.add_argument('--num-class', type = int, default = 8)
	parser.add_argument('--text-len', type = int, default = 128)
	parser.add_argument('--image-height', type = int, default = 224)
	parser.add_argument('--image-width', type = int, default = 224)
	parser.add_argument('--embedding-dim', default = None) 
	parser.add_argument('--batch-size', type = int, default = 1)
	parser.add_argument('--vocab-size', default = None)
	parser.add_argument('--vocabs', type = str, default = 'dataset/vocabs.txt')
	parser.add_argument('--test-data', type = str, default = 'dataset/test.tfrecords')
	parser.add_argument('--pretrained-embedding', type = str, default = 'glove.twitter.27B/glove.twitter.27B.100d.txt')
	main(parser.parse_args())
