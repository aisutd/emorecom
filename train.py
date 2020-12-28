"""
train.py - training module
"""

# import dependencies
import os
import glob
import argparse
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks, losses, metrics

# import local packages
from emorecom.data import Dataset
from emorecom.model import create_model

# get directory path
DIR_PATH = os.getcwd()

def main(args):

	"""------parser-arugments------"""
	# initialize dataset
	assert args.train_data
	TRAIN_DATA = os.path.join(DIR_PATH, args.train_data)
	if args.test_data:
		TEST_DATA = os.path.join(DIR_PATH, args.test_data)
	
	# initialize train-dataset
	print("Creating Data Loading")
	VOCABS = os.path.join(DIR_PATH, args.vocabs)
	dataset = Dataset(data = TRAIN_DATA, vocabs = VOCABS, image_size = [args.image_height, args.image_width],
		text_len = args.text_len, batch_size = args.batch_size)
	train_data = dataset(training = True)

	# test train-dataset
	#images, transcripts, labels = next(iter(train_data))
	#print(labels)
	#print(images.shape, transcripts.shape, labels.shape)
	for sample in train_data.take(1):
		features, labels = sample
		print(features['image'].shape, features['transcripts'].shape, labels)

	# initialize model
	print("Initialize and compile model")
	MODEL_CONFIGS= {'img_shape' : [args.image_height, args.image_width, 3],
		'text_len' : args.text_len,
		'vocabs' : VOCABS,
		'vocab_size' : args.vocab_size,
		'embed_dim' : args.embedding_dim,
		'pretrained_embed' : os.path.join(DIR_PATH, args.pretrained_embedding),
		'num_class' : args.num_class}
	model = create_model(configs = MODEL_CONFIGS)
	print(model.summary())

	# set hyperparameters
	OPTIMIZER = optimizers.Adam(learning_rate = args.learning_rate)
	LOSS = losses.CategoricalCrossentropy(from_logits = False)
	METRICS = [metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()]

	# compile model
	model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)

	# set hyperparameters
	print("Start training")
	LOG_DIR = os.path.join(DIR_PATH, args.logdir, args.experiment_name)
	CHECKPOINT_PATH = os.path.join(DIR_PATH, args.checkpoint_dir, args.experiment_name)
	CALLBACKS = [
		callbacks.TensorBoard(log_dir = LOG_DIR, write_images = True),
		callbacks.ModelCheckpoint(filepath = CHECKPOINT_PATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')]
	STEPS_PER_EPOCH = None
	model.fit(train_data, verbose = 1, callbacks = CALLBACKS, epochs = args.epochs,
		steps_per_epoch = STEPS_PER_EPOCH)

	# save model
	#model_path = os.path.join(DEFAULT_DIR, args.models, args.experiment_name)
	#tf.saved_model.save(model, model_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	# add arguments
	parser.add_argument('--experiment-name', type = str, default = 'model')
	parser.add_argument('--num-class', type = int, default = 8)
	parser.add_argument('--text-len', type = int, default = 128)
	parser.add_argument('--image-height', type = int, default = 224)
	parser.add_argument('--image-width', type = int, default = 224)
	parser.add_argument('--embedding-dim', default = None) 
	parser.add_argument('--batch-size', type = int, default = 1)
	parser.add_argument('--learning-rate', type = float, default = 0.0001)
	parser.add_argument('--epochs', type = int, default = 1)
	parser.add_argument('--vocab-size', default = None)
	parser.add_argument('--vocabs', type = str, default = 'dataset/vocabs.txt')
	parser.add_argument('--train-data', type = str, default = 'dataset/train.tfrecords')
	parser.add_argument('--test-data', type = str, default = None)
	parser.add_argument('--validation-data', type = str, default = 'dataset/validation.tfrecords')
	parser.add_argument('--logdir', type = str, default = 'logs')
	parser.add_argument('--checkpoint-dir', type = str, default = 'checkpoints')
	parser.add_argument('--pretrained-embedding', type = str, default = 'glove.twitter.27B/glove.twitter.27B.100d.txt')
	parser.add_argument('--output', type = str, default = 'models')
	main(parser.parse_args())
