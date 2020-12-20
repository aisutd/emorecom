"""
preprocess.py - training module
"""

# import dependencies
import os
import json
import glob
import argparse
import pandas as pd
import tensorflow as tf

# default path
DEFAULT_PATH = os.path.join(os.getcwd(), 'dataset')
EMOTIONS = ['angry', 'disgust', 'feear', 'happy', 'sad', 'surprise', 'neutral', 'others']

def train_concat(file_name, image_path, transcripts, labels):
	"""
	concat - function to concat images, transcripts, and labels together
	"""

	def parse(image, transcript, label):
		output = {
			'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
			'transcripts' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [transcript])),
			'label' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [label]))}
		
		output = tf.train.Example(features = tf.train.Features(feature = output)).SerializeToString()

		# check file
		print(tf.train.Example.FromString(output))
		return output

	with tf.io.TFRecordWriter(file_name) as writer:
		for transcript in transcripts:
			try:
				# retrieve labels, image, and transcripts
				print(labels[labels['image_id'] == transcript['img_id']])
				label = list(
					labels[labels['image_id'] == transcript['img_id']]
						.iloc[0][EMOTIONS])
				label = ','.join([str(x) for x in label]).encode('utf-8') # convert to string
				texts = ';'.join(transcript['dialog']).encode('utf-8')
				img = os.path.join(image_path, transcript['img_id'] + '.jpg').encode('utf-8')

				# parse image, transcript, label to tfrecord-example
				writer.write(parse(img, texts, label))
			except Exception as e:
				print(e)

def test_concat(file_name, image_path, transcripts):
        """
        concat - function to concat images, transcripts, and labels together
        """

        def parse(input):
                output = {
                        'input' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [input]))
                }
                output = tf.train.Example(features = tf.train.Features(feature = output)).SerializeToString()

                # check file
                return output

        with tf.io.TFRecordWriter(file_name) as writer:
                for transcript in transcripts:
                        try:
                                # retrieve labels, image, and transcripts
                                texts = ';'.join(transcript['dialog'])
                                img = os.path.join(image_path, transcript['img_id'] + '.jpg')

                                # concat img, transcripts, and label
                                concat = '/'.join([img, texts]).encode('utf-8')

                                # parse image, transcript, label to tfrecord-example
                                writer.write(parse(concat))
                        except Exception as e:
                                print(e)
def test(filename):
	"""
	test - function to inspect if data is concatonated correctly
	"""

	# read tfrecord file
	data = tf.data.TFRecordDataset(filename)

	for sample in data.take(5):
		print(sample)

	def _parse(input):
		feature_details = {
			'image' : tf.io.FixedLenFeature([], tf.string),
			'transcripts' : tf.io.FixedLenFeature([], tf.string),
			'label' : tf.io.FixedLenFeature([], tf.string)}
		return tf.io.parse_single_example(input, feature_details)

	#parsed_data = data.map(lambda x: tf.io.parse_single_example(x, feature_details))
	parsed_data = data.map(_parse)
	print(next(iter(parsed_data)))

def main(args):
	
	# initialize train dataset
	transcripts = os.path.join(DEFAULT_PATH, args.transcript)
	image_path = os.path.join(DEFAULT_PATH, args.image)
	labels = os.path.join(DEFAULT_PATH, args.label)
	output = os.path.join(DEFAULT_PATH, args.output)

	# read transcripts
	with open(transcripts) as file:
		transcripts = json.load(file)

	# read labels
	labels = pd.read_csv(labels)
	# rename columns
	labels = labels.rename(
		columns = {
			old:new for old, new in zip(labels.columns, ['id', 'image_id'] + EMOTIONS)})

	# concat images, transcripts, and labels (if training is True)
	if args.training:
		train_concat(output, image_path, transcripts, labels)
		test(output)
	else:
		test_concat(output, image_path, transcripts)

	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Argument Parser')

	# add arguments
	parser.add_argument('--training', type = bool, default = True)
	parser.add_argument('--image', type = str, default = os.path.join('warm-up-train', 'train'))
	parser.add_argument('--transcript', type = str, default = os.path.join('warm-up-train', 'train_transcriptions.json'))
	parser.add_argument('--label', type = str, default = os.path.join('warm-up-train', 'train_emotion_labels.csv'))
	parser.add_argument('--output', type = str, default = os.path.join('train.tfrecords'))
	main(parser.parse_args())
