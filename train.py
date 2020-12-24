"""
train.py - training module
"""

# import dependencies
import os
import glob
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks, losses, metrics

# import local packages
from emorecom.data import Dataset
from emorecom.model import create_model

def main():
	# path variables
	LOG_DIR = os.path.join(os.getcwd(), 'logs')
	CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

	# initialize experiment-name
	experiment = 'model-0'

	# initialize train dataset
	train_path = os.path.join(os.getcwd(), 'dataset', 'train.tfrecords')
	
	# initialize train-dataset
	print("Creating Data Loading")
	vocab_path = os.path.join(os.getcwd(), 'dataset', 'vocabs.txt')
	max_len = 128
	batch_size = 4
	dataset = Dataset(
		data_path = train_path, vocabs = vocab_path, 
		max_len = max_len, batch_size = batch_size)
	train_data = dataset(training = True)

	# test train-dataset
	for sample in train_data.take(1):
		images, transcripts, labels = sample
		print(images.shape, transcripts.shape, labels.shape)
		input()

	# initialize model
	print("Initialize and compile model")
	MODEL_CONFIGS= {
		'img_shape' : [224, 224, 3],
		'text_shape' : [50],
		'vocabs' : vocab_path,
		'vocab_size' : None,
		'max_len' : max_len,
		'embed_dim' : None,
		'pretrained_embed' : './glove.twitter.27B/glove.twitter.27B.100d.txt',
		'num_class' : 8}
	#model = create_model(configs = MODEL_CONFIGS)
	#print(model.summary())

	# set hyperparameters
	# compile model
	LR = 0.0001
	OPTIMIZER = optimizers.Adam(learning_rate = LR)
	LOSS = losses.CategoricalCrossentropy(from_logits = True)
	METRICS = [metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()]
	#model.compile(optimizer = OPTIMZER, loss = LOSS, metrics = METRICS)

	# set hyperparameters
	print("Start training")
	LOG_DIR = os.path.join(LOG_DIR, experiment)
	CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, experiment)
	CALLBACKS = [
		callbacks.TensorBoard(log_dir = LOG_DIR, write_images = True),
		callbacks.ModelCheckpoint(filepath = CHECKPOINT_PATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')]
	EPOCHS = 50
	STEPS_PER_EPOCH = None
	#model.fit(train_data, verbose = 1, callbacks = CALLBACKS, epochs = EPOCHS,
	#	steps_per_epoch = STEPS_PER_EPOCH)

	# save model
	#model_path = experiment
	#tf.saved_model.save(model, model_path)

if __name__ == '__main__':
	main()
