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
from theano.tensor.shared_randomstreams import RandomStreams


options = {
	"batch_size" : 1,
	"lr" : 0.1,
	"cd_k" : 15,
	"n_hidden" : 100,
	"print_freq" : 50,
	"valid_freq" : 50,
	"n_epoch" : 100,
	"optimizer" : "adadelta"
}

optimizer = {"sgd" : optimization.sgd, 
			"momentum" : optimization.momentum, 
			"nesterov_momentum" : optimization.nesterov_momentum, 
			"adagrad" : optimization.adagrad,
			"adadelta" : optimization.adadelta,
			"rmsprop" : optimization.rmsprop}


