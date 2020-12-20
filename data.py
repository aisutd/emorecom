"""
data.py - data-loading module
"""

# import dependencies
import os
import tensorflow as tf
import cv2

# import local packages


class Dataseet:
	"""
	Dataset - class to implement Tensorflow Data API
	"""

	def __init__(self, image, transcript, label = None, batch_size = 1):
		"""
		Class constructor:
		Inputs:
			- image : str
				Path to image folder
			- transcript : str
				Path to transcript file
			- label : str
				Path to label file
		"""

		# retrieve image paths 
		self.images = [os.path.join(image, img) for img in os.listdir(image_path)]

		# read transcripts
		with open(transcript) as file:
			self.transcripts = self.preprocess_transcript(json.load(file))


		# read label file
		if label:
			self.labels = pd.read_csv(label_path)

	def preprocess_transcript(self, input):
		return input

	def __call__(self, training = False):
		return None
