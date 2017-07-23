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

from __future__ import print_function
import os, sys, glob
import time
import itertools
import nltk
import numpy
import pickle
from nltk.corpus import stopwords
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T 
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import optimization

options = {
	"feature_map" : 100,
	"filter_h" : [3,4,5],
	"classes" : 2,
	"wordvec" : 300,
	"dropout" : [0.8,0.8],
	"batch_size" : 100,	
	"validation" : 0.1,    # validation set 
	"print_freq" : 50,
	"valid_freq" : 50,
	"n_epoch" : 30,
	"optimizer" : "adadelta"
}


optimizer = {"sgd" : optimization.sgd, 
			"momentum" : optimization.momentum, 
			"nesterov_momentum" : optimization.nesterov_momentum, 
			"adagrad" : optimization.adagrad,
			"adadelta" : optimization.adadelta,
			"rmsprop" : optimization.rmsprop}

