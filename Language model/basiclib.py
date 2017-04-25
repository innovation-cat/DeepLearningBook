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
#    n_words:  the number of input layer units
#    n_emb:  the number of embedding layer units
#    n_hidden:  the number of hidden layer units
#    n_output:  the number of output layer units 
#    batch_size: the number of training set of each iteration   
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================


import numpy
import theano
import theano.tensor as T 
import os, sys, time
import itertools 
import csv
import glob
import nltk
from collections import *
from nltk.corpus import stopwords



options = {'n_words':20001, 
			'n_emb':128, 
			'n_hidden':128, 
			'n_output':20001, 
			'batch_size':500}

			