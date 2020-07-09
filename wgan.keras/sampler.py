import tensorflow as tf
import os
import random
import numpy as np
import cv2

class DataSampler(object):
    def __init__(self, data_directory, large_dataset=False):
        self.X = []
        self.large_dataset = large_dataset
        self.head = 0

        for dirname, _, filenames in os.walk(data_directory):
            for filename in filenames:
                input_path = os.path.join(dirname, filename)
                if(large_dataset):
                	self.X.append(input_path)
                else:
                	self.X.append(cv2.resize(cv2.imread(input_path), (64, 64)))

    def sample(self, batch_size):
    	tail = min(self.head + batch_size, len(self.X))
    	if(self.large_dataset):
    		X_batch = np.array([cv2.resize(cv2.imread(filename), (64, 64)) for filename in self.X[self.head : tail]]) / 255.0
    	else:
    		X_batch = np.array(self.X[self.head : tail]) / 255.0
    	self.head = tail % len(self.X)
    	return X_batch

class NoiseSampler(object):
    def __init__(self, num_features):
        self.num_features = num_features

    def sample(self, batch_size):
        noise = tf.random.normal(shape=[batch_size, self.num_features])
        return noise


# dataSampler = DataSampler('images', False)

# for i in range(30):
# 	print(dataSampler.head, dataSampler.sample(32).shape)