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
#    n_words:  input layer
#    n_emb:  embedding layer 
#    n_hidden:  hidden layer
#    n_output:  output layer 
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

			