# coding: utf-8
#
# sda.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of stacked denoising autoencoder
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function, division
from basiclib import *
from theano.tensor.shared_randomstreams import RandomStreams
from softmax import *
from mlp import *

class DA:
	def __init__(self, input, n_input, n_hidden, W=None, bhid=None, bout=None):
		self.input = input
		self.n_input = n_input
		self.n_output = n_input
		self.n_hidden = n_hidden 
		
		#print(type(W))
		if W is None:
			initial_W = numpy.random.uniform(
					low=-4*numpy.sqrt(6. / (n_hidden + n_input)),
                    high=4*numpy.sqrt(6. / (n_hidden + n_input)),
                    size=(n_input, n_hidden)).astype(theano.config.floatX)
			W = theano.shared(value = initial_W, name = 'W')
		self.W = W
		
		if bhid is None:
			initial_bhid = numpy.zeros(shape=(n_hidden, )).astype(theano.config.floatX)
			bhid = theano.shared(value = initial_bhid, name = 'bhid')
		self.bhid = bhid
		
		if bout is None:
			initial_bout = numpy.zeros(shape=(n_input, )).astype(theano.config.floatX)
			bout = theano.shared(value = initial_bout, name = 'bout')
		self.bout = bout

		self.W_pi = self.W.T
		self.params = [self.W, self.bhid, self.bout]
		self.hidden = self.get_hidden_value(self.input)
		self.output = self.get_reconstructed_value(self.hidden)
		
		self.theano_rng = RandomStreams(12345)
	
	def get_corrupted_input(self, input, corrupted_level):
		return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corrupted_level,
                                        dtype=theano.config.floatX) * input	
		
	def get_hidden_value(self, x):
		return T.nnet.sigmoid(T.dot(x, self.W)+self.bhid)
		
	def get_reconstructed_value(self, x):
		out = T.nnet.sigmoid(T.dot(x, self.W_pi)+self.bout)
		return out
		
	def get_cost_update(self, lr, reg, corrupted_level):
		x = self.get_corrupted_input(self.input, corrupted_level)
		#x = self.input
		y = self.get_hidden_value(x)
		z = self.get_reconstructed_value(y)
		w, h = self.input.shape
		#cost = (T.sum((self.input-z)**2))/(w*h*1.0) + reg*((self.W**2).sum())
		#cost = T.sum((self.input-z)**2, axis=1)
		cost = -T.sum(self.input*T.log(z) + (1-self.input)*(T.log(1-z)), axis=1)
		cost = T.mean(cost)
		#cost = T.mean(T.sum((self.input-z)**2))
		gparams = T.grad(cost, self.params)
		updates = [(p, p-lr*gp) for p, gp in zip(self.params, gparams)]
		return (cost, updates)
		

class SDA:
	def __init__(self, input, n_input, n_hiddens, n_output):
		self.input = input
		self.da_layers = []
		self.hidden_layers = []
		
		layers_size = [n_input] + n_hiddens + [n_output]
		weight_matrix_size = zip(layers_size[:-1], layers_size[1:])
		data = input
		
		for n_in, n_out in weight_matrix_size[:-1]:
			hidden_layer = HiddenLayer(data, n_in, n_out)
			da_layer = DA(data, n_in, n_out, W=hidden_layer.W, bhid=hidden_layer.b)
			self.da_layers.append(da_layer)
			self.hidden_layers.append(hidden_layer)
			data = self.hidden_layers[-1].output
		
		n_in, n_out = weight_matrix_size[-1]
		self.output_layer = SoftmaxLayer(data, n_in, n_out)
			
		self.params = list(itertools.chain(*[hidden.params for hidden in self.hidden_layers]))
		self.params.append(self.output_layer.W)
			
	def get_cost_updates(self, y, lr, reg, optimizer_fun):
		cost = self.output_layer.cross_entropy(y)
		sum_W = 0.0
		for hidden in self.hidden_layers:
			sum_W = sum_W + ((hidden.W)**2).sum()
		sum_W = sum_W + ((self.output_layer.W)**2).sum()
		cost = cost + reg*sum_W
		try:
			updates = optimizer_fun(cost, self.params, lr)
		except:
			print("Error: no optimizer function")
		else:
			return (cost, updates)
		
	def error_rate(self, y):
		return self.output_layer.error_rate(y)
			
if __name__ == "__main__":
	train_x, train_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/*_batch*")
	
	
	#valid_x, valid_y = (train_x[40000:], train_y[40000:])
	train_x, train_y = (train_x[0:40000], train_y[0:40000])
	
	_min, _max = float(numpy.min(train_x)), float(numpy.max(train_x))
	train_x = ((train_x - _min) / (1.0*(_max - _min))).astype(theano.config.floatX)

	
	test_x, test_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/test_batch")
	#test_x, test_y = (test_x, test_y)
	test_x = ((test_x - _min) / (1.0*(_max - _min))).astype(theano.config.floatX)
	
	#print(train_x.dtype)
	
	train_set_size, col = train_x.shape
	#valid_set_size, _ = valid_x.shape
	test_set_size, _ = test_x.shape
	
	x = T.matrix('x').astype(theano.config.floatX)
	y = T.ivector('y')
	corrupted_level = T.scalar('cl').astype(theano.config.floatX)
	index = T.scalar('lr', dtype='int64')
	lr = T.scalar('lr', dtype=theano.config.floatX)
	reg = T.scalar('reg', dtype=theano.config.floatX)
	
	batch_size = options['batch_size']
	n_train_batch = train_set_size//batch_size
	#n_valid_batch = valid_set_size//batch_size
	n_test_batch = test_set_size//batch_size
	
	shared_train_x, shared_train_y = shared_data(train_x, train_y)
	
	model = SDA(x, col, [1000, 1000], 10)
	
	for da in model.da_layers:
		print(da.n_input, da.n_hidden, da.n_output)
	
	#print(optimizer[options["optimizer"]])
	cost, updates = model.get_cost_updates(y, lr, reg, optimizer[options["optimizer"]])

	train_model = theano.function(inputs = [x, y, lr, reg], outputs = cost, updates = updates, on_unused_input = 'ignore')
	
	train_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	#valid_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	test_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
		
	pretraining_functions = []
	pretrain_batch_size = 5000
	n_train_pre = train_set_size//pretrain_batch_size
	for da in model.da_layers:
		cost, updates = da.get_cost_update(lr, reg, corrupted_level)
		pretraining_functions.append(theano.function(
			inputs = [index, lr, reg, theano.In(corrupted_level, value=0.1)],
			outputs = cost, updates = updates, 
			givens = {model.input : shared_train_x[index*pretrain_batch_size:(index+1)*pretrain_batch_size]},
			on_unused_input = 'ignore'
		))
		
	idx = numpy.arange(train_set_size)
	
	for p in model.params:
		print("%lf "%numpy.mean(p.get_value()), end="")
	print("\n")
	
	for id, da in enumerate(model.da_layers):
		print(id)
		for e in range(200):
			c = []
			numpy.random.shuffle(idx)
			new_train_x = [train_x[i] for i in idx]
			for n_batch_index in range(n_train_pre):
				c.append(pretraining_functions[id](n_batch_index, 0.001, 0.0, 0.0))
			print('Pre-training layer %d, epoch %d, cost %f' % (id, e, numpy.mean(c, dtype='float64')))
		
	for p in model.params:
		print("%lf "%numpy.mean(p.get_value()), end="")
	print("\n")
	
	train_num = 0
	best_err = 1.0
	error_output = open("sda.txt", "wb")
	with open("model_sda.npz", "wb") as fout:
		for epoch in range(options["n_epoch"]):
			numpy.random.shuffle(idx)
			new_train_x = [train_x[i] for i in idx]
			new_train_y = [train_y[i] for i in idx]
			for n_batch_index in range(n_train_batch):
				c = train_model(
					new_train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					new_train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					0.1, 0.0
				)
				train_num = train_num + 1
				if train_num%options["print_freq"]==0:
					print("train num: %d, cost: %lf"%(train_num, c))
				
				if train_num%options["valid_freq"]==0:
					train_errors = [train_err(train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 0.1, 0.0) for n_batch_index in range(n_train_batch)]
					
					test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.1, 0.0) for n_test_index in range(n_test_batch)]
					
					if numpy.mean(test_errors) < best_err:
						best_err = numpy.mean(test_errors)
						
						
						params = dict([(p.name, p.get_value()) for p in model.params])
						numpy.savez(fout, params)
						
						print("train num: %d, best train error: %lf, best test error: %lf"%(train_num, numpy.mean(train_errors), numpy.mean(test_errors)))
						
			print("epoch %d end"%epoch)
			test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.1, 0.0) for n_test_index in range(n_test_batch)]
			print("%lf"%numpy.mean(test_errors), file=error_output)
	
	