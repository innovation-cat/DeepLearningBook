# coding: utf-8
#
# basiclib.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of convolutional neural network for sentiment analysis.
#
# 	 citation:
#    	1: Yoon Kim. Convolutional Neural Networks for Sentence Classification. EMNLP 2014.
#		2: N Kalchbrenner, E Grefenstette, P Blunsom. A Convolutional Neural Network 
#          for Modelling Sentences. ACL, 2014.
# 
# optimizer: optimization algorithm
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function
from basiclib import *
from cnn_model import *


#def adadelta():
	
def get_idx_for_word(data, word2idx, max_len):
	ret = []
	pad = 4
	for sent in data:
		x = []
		for i in range(pad):
			x.append(0)
		for word in sent:
			if word in word2idx:
				x.append(word2idx[word])
		while len(x) < max_len+2*pad:
			x.append(0)
		ret.append(x)
	return ret

if __name__ == "__main__":
	print("start to load dataset, %s" % time.strftime("%Y-%m-%d : %X", time.localtime()))
	with open("imdb.pkl", "rb") as fin:
		train_set, test_set, vocab, word2idx, word2vec = cPickle.load(fin)
	print("end to load dataset, %s\n" % time.strftime("%Y-%m-%d : %X", time.localtime()))

	
	print("start to build model, %s" % time.strftime("%Y-%m-%d : %X", time.localtime()))
	train_set_x, train_set_y = train_set
	test_set_x, test_set_y = test_set
	#print("length of train set: %d, length of test set: %d" % (len(train_set_x), len(test_set_x)))
	max_len = numpy.max([len(sent) for sent in train_set_x])
	min_len = numpy.min([len(sent) for sent in train_set_x])
	#print(max_len, min_len)
	
	train_set_x = get_idx_for_word(train_set_x, word2idx, max_len)
	test_set_x = get_idx_for_word(test_set_x, word2idx, max_len)

	img_h = len(train_set_x[0])
	img_w = options["wordvec"]
	
	filters_h = options["filter_h"]
	filter_w = options["wordvec"]
	feature_maps = options["feature_map"]
	
	filter_shapes = []
	pool_sizes = []
	for filter_h in filters_h:
		filter_shapes.append((100, 1, filter_h, filter_w))
		pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
	
	#print("image shape, w: %d, h: %d" % (img_w, img_h))
	#print("filter shape, w: %d" % (filter_w))
	#print("feature maps: %d" % feature_maps)
	batch_size = options["batch_size"]
	x = T.matrix('x', dtype='int64')
	y = T.ivector('y')
	words = theano.shared(value=word2vec, name='word2vec')
	conv_input = words[x.flatten()].reshape((batch_size, 1, img_h, img_w))
	
	rng = numpy.random.RandomState(12345)
	full_inputs, conv_layers = [], []
	
	for idx, filter_h in enumerate(filters_h):
		conv_layer = LeNetConvPoolLayer(
			rng, input=conv_input, image_shape=(batch_size, 1, img_h, img_w),
			filter_shape=filter_shapes[idx],
			poolsize=pool_sizes[idx]
		)
		full_input = conv_layer.output.flatten(2)
		conv_layers.append(conv_layer)
		full_inputs.append(full_input)
	full_input=T.concatenate(full_inputs, 1)
	hidden_layers = [100]
	model = MLPDropout(rng, full_input, 300, hidden_layers, 2, options["dropout"])
	print("end to build model, %s\n" % time.strftime("%Y-%m-%d : %X", time.localtime()))
	
	params = model.params
	for layer in conv_layers:
		params = params + layer.params
		
	cost = model.cross_entropy(y)
	dropout_cost = model.dropout_cross_entropy(y)
	
	#gparams = T.grad(dropout_cost, params)
	
	lr = T.scalar('lr', dtype=theano.config.floatX)

	updates = optimizer[options["optimizer"]](dropout_cost, params, lr)
	
	train_set_size = len(train_set_x)
	test_set_size = len(test_set_x)
	
	# ignore validation set
	#valid_set_x = train_set_x[:int(train_set_size*options["validation"])]
	#valid_set_y = train_set_y[:int(train_set_size*options["validation"])]
	
	#train_set_x = train_set_x[int(train_set_size*options["validation"]):]
	#train_set_y = train_set_y[int(train_set_size*options["validation"]):]
	
	#train_set_size = len(train_set_x)
	#valid_set_size = len(valid_set_x)
	
	print("train size: %d, test size: %d\n"%(train_set_size, test_set_size))
	
	
	n_train_batch = int(numpy.round(((train_set_size*1.0)/(batch_size*1.0))+0.5))
	n_test_batch = int(numpy.round(((test_set_size*1.0)/(batch_size*1.0))+0.5))
	idxs = numpy.arange(train_set_size)
	
	
	train_model_fun = theano.function(inputs = [x, y, lr], outputs = cost, updates = updates, on_unused_input='ignore')
	
	train_error_fun = theano.function(inputs = [x, y, lr], outputs = model.errors(y), on_unused_input='ignore')
	test_error_fun = theano.function(inputs = [x, y, lr], outputs = model.errors(y), on_unused_input='ignore')
	
	print_freq = options["print_freq"]
	valid_freq = options["valid_freq"]
	
	train_num = 0
	
	fout = open("model_adadelta.txt", "wb")
	best_error = 1.0
	print("start to train model: %s"%time.strftime("%Y-%m-%d, %X", time.localtime()))
	for learning_rate in numpy.linspace(0.1, 0.1, 1).astype(theano.config.floatX):
		for epoch in range(options["n_epoch"]):
			numpy.random.shuffle(idxs)
			new_train_set_x = [train_set_x[id] for id in idxs]
			new_train_set_y = [train_set_y[id] for id in idxs]
			avg_cost = []
			for train_batch in range(n_train_batch):
				if train_batch==n_train_batch-1:
					break
				c = train_model_fun(new_train_set_x[train_batch*batch_size : (train_batch+1)*batch_size], new_train_set_y[train_batch*batch_size : (train_batch+1)*batch_size], learning_rate)
				
				avg_cost.append(c)
				train_num = train_num + 1
				
				if train_num%print_freq==0:
					print("train num: %d, cost: %lf"%(train_num, c))
				if train_num%valid_freq==0:
					train_error, test_error = [], []
					for train_batch in range(n_train_batch):
						if train_batch==n_train_batch-1:
							break
						train_error.append(train_error_fun(new_train_set_x[train_batch*batch_size : (train_batch+1)*batch_size], new_train_set_y[train_batch*batch_size : (train_batch+1)*batch_size], learning_rate))
						
					for test_batch in range(n_test_batch):
						if test_batch==n_test_batch-1:
							break
						test_error.append(test_error_fun(test_set_x[test_batch*batch_size : (test_batch+1)*batch_size], test_set_y[test_batch*batch_size : (test_batch+1)*batch_size], learning_rate))
					
					if numpy.mean(test_error) < best_error:
						best_error = numpy.mean(test_error)
						
						
						pp = dict([(p.name, p.get_value()) for p in params])
						numpy.savez(fout, pp)
						
						print("train num: %d, best train error: %lf, best test error: %lf"%(train_num, numpy.mean(train_error), numpy.mean(test_error)))
	
	print("end to train model: %s\n"%time.strftime("%Y-%m-%d, %X", time.localtime()))

	