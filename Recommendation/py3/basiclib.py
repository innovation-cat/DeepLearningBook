# coding: utf-8
#
# basiclib.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Basic Configuration and Interface
# 
# options: hyper-parameter setting
#
# optimizer: optimization algorithm
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

import os 
import sys
import numpy
import theano
import time
import itertools
import glob, gzip
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
from collections import *
import optimization
from theano.tensor.shared_randomstreams import RandomStreams

if sys.version_info[0]==3:
	import pickle
else:
	import cPickle


options = {
	"batch_size" : 1,
	"lr" : 0.05,
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


