# coding: utf-8
#
# basiclib.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Basic Configuration and Interface
# 
# options: hyper-parameter setting, you should change the value according to different model:
#		Softmax: Softmax Regression 
#			sgd: lr: 0.00000001,  reg: 0.0, epoch: 1000, batch_size: 500  => accuracy: 0.4
#			adadelta: lr: 0.0001,  reg: 0.0, epoch: 1000, batch_size: 500  => accuracy: 0.4
# 		MLP: Multi-Layer Perceptron 
#			lr: 0.0001, reg: 0.01, epoch: 1000, batch_size: 100  =>  accuracy: 0.5
# 		SDA: Stacked Denoising Autoencoder: 
#			lr: 0.0001, reg: 0.01, pre-training epoch: 200, fine-tune epoch: 1000, batch_size: 100  =>  accuracy: 0.5
#
# load_cifar10_dataset: load cifar10 dataset function, copy from: http://www.cs.toronto.edu/~kriz/cifar.html
#
# load_mnist_dataset: load mnist dataset function.
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

import os 
import sys
import numpy
import theano
import time
import cPickle
import itertools
import glob, gzip
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
from collections import *
import optimization


options = {
	"batch_size" : 100,
	"lr" : 0.0001,
	"reg" : 0.01,
	"n_output" : 10,
	"print_freq" : 50,
	"valid_freq" : 50,
	"n_epoch" : 300,
	"optimizer" : "adadelta"
}

optimizer = {"sgd" : optimization.sgd, 
			"momentum" : optimization.momentum, 
			"nesterov_momentum" : optimization.nesterov_momentum, 
			"adagrad" : optimization.adagrad,
			"adadelta" : optimization.adadelta,
			"rmsprop" : optimization.rmsprop}

def unpickle(file):
    with open(file, 'rb') as fin:
		dict = cPickle.load(fin)
    return dict
   
def shared_data(data_x, data_y):

	shared_data_x = theano.shared(
		value=data_x.astype(theano.config.floatX), name = 'shared_x', borrow = True
	)
	
	shared_data_y = theano.shared(
		value=numpy.asarray(data_y, dtype=theano.config.floatX), name = 'shared_x', borrow = True
	)
	
	return shared_data_x, T.cast(shared_data_y, "int32")
	
def load_cifar10_dataset(dir):
	train_x, train_y = [], []
	for file in glob.iglob(dir):
		data = unpickle(file)
		data_x, data_y = data['data'], data['labels']
		data_y = numpy.asarray(data_y)
		train_x.append(data_x)
		train_y.append(data_y)
	train_x = numpy.concatenate(train_x)
	train_y = numpy.concatenate(train_y)
	return train_x, train_y

	
def load_mnist_dataset(dir):
	with gzip.open(dir, 'rb') as f:
		try:
			train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
		except:
			train_set, valid_set, test_set = cPickle.load(f)
			
	shared_train_set = shared_data(train_set[0], train_set[1])
	shared_test_set = shared_data(test_set[0], test_set[1])
	
	return shared_train_set, shared_test_set
