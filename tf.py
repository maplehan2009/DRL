from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

class my_network:
	def __init__(self):
		# Specify that all features have real-value data
		feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20], n_classes=3, model_dir="/homes/jh1016/DRL/toy/iris_model")
		
	def train(self, X_train, y_train):
		# Define the training inputs
		def get_train_inputs():
			x = tf.constant(X_train)
			y = tf.constant(y_train)
			return x, y
			
		# Fit model.
		self.classifier.fit(input_fn=get_train_inputs, steps=500)

	def evaluate(self, X_test, y_test):
		# Define the test inputs
		def get_test_inputs():
			x = tf.constant(X_test)
			y = tf.constant(y_test)
			return x, y

		# Evaluate accuracy.
		accuracy_score = self.classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
		return accuracy_score

def get_data():
	# If the training and test sets aren't stored locally, download them.
	if not os.path.exists(IRIS_TRAINING):
		raw = urllib.urlopen(IRIS_TRAINING_URL).read()
		with open(IRIS_TRAINING, "w") as f:
		  f.write(raw)

	if not os.path.exists(IRIS_TEST):
		raw = urllib.urlopen(IRIS_TEST_URL).read()
		with open(IRIS_TEST, "w") as f:
		  f.write(raw)

	# Load datasets.
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)
	X_train = training_set.data
	y_train =training_set.target
	X_test = test_set.data
	y_test = test_set.target
	return X_train, y_train, X_test, y_test
	
if __name__ == "__main__":
	X_train, y_train, X_test, y_test = get_data()
	N = len(X_train)
	n = int(N/4)
	my_nn = my_network()
	my_nn.train(X_train[:n], y_train[:n])
	acc1 = my_nn.evaluate(X_test, y_test)
	my_nn.train(X_train[n:], y_train[n:])
	acc2 = my_nn.evaluate(X_test, y_test)
	print(acc1, acc2)
	

