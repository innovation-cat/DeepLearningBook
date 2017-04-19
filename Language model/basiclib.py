
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
			'batch_size':50}
