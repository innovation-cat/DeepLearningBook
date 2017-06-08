# coding: utf-8
#
# mlp.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of multilayer perceptron
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function
from basiclib import *
from softmax import *

class HiddenLayer:
	def __init__(self, input, n_input, n_output, W=None, b=None, activation=T.nnet.relu):
		self.input = input
		self.n_input = n_input
		self.n_output = n_output

		# 权重初始化采用经典的 Xavier initialization
		if W is None:
			W = numpy.random.uniform(
				low = -numpy.sqrt(6.0/(n_input+n_output)),
				high = numpy.sqrt(6.0/(n_input+n_output)),
				size = (n_input, n_output)).astype(theano.config.floatX)
		self.W = theano.shared(value=W, name='W', borrow=True)

		if b is None:
			b = numpy.zeros(shape=(n_output, )).astype(theano.config.floatX)
		self.b = theano.shared(value=b, name='b', borrow=True)
		
		self.params = [self.W, self.b]
		self.output = activation(T.dot(input, self.W)+self.b)
		
class MLP:
	def __init__(self, input, n_input, n_hiddens, n_output, activation=T.nnet.relu):
		self.input = input
		
		self.n_input = n_input 
		self.n_hiddens = n_hiddens
		self.n_output = n_output 
		
		self.hidden_layers = []
		layers = [n_input] + n_hiddens + [n_output]
		weight_matrix_size = zip(layers[:-1], layers[1:])
		data = input 
		for n_in, n_out in weight_matrix_size[:-1]:
			self.hidden_layers.append(HiddenLayer(data, n_in, n_out))
			data = self.hidden_layers[-1].output
		
		n_in, n_out = weight_matrix_size[-1]
		self.output_layer = SoftmaxLayer(data, n_in, n_out)
		
		self.params = []
		for hidden in self.hidden_layers:
			self.params = self.params + hidden.params
		self.params = self.params + self.output_layer.params
		
	def get_cost_updates(self, y, lr, reg, optimizer_fun):
		cost = self.output_layer.cross_entropy(y)
		L = 0.0
		for hidden in self.hidden_layers:
			L=L+(hidden.W**2).sum()
		L=L+(self.output_layer.W**2).sum()
		
		cost = cost + reg*L
		try:
			updates = optimizer_fun(cost, self.params, lr)
		except:
			print("Error: no optimizer function")
		else:
			return (cost, updates)
		
	def error_rate(self, y):
		return self.output_layer.error_rate(y)

	
if __name__ == "__main__":
	train_x, train_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/data_batch_*")
	test_x, test_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/test_batch")
	
	train_x = train_x / 255.0
	test_x = test_x / 255.0
	
	
	train_set_size, col = train_x.shape
	test_set_size, _ = test_x.shape
	
	x = T.matrix('x').astype(theano.config.floatX)
	y = T.ivector('y')
	index = T.iscalar('index')
	lr = T.scalar('lr', dtype=theano.config.floatX)
	reg = T.scalar('reg', dtype=theano.config.floatX)
	
	batch_size = options['batch_size']
	n_train_batch = train_set_size//batch_size
	n_test_batch = test_set_size//batch_size
	
	model = MLP(x, col, [1000, 1000], 10)
	cost, updates = model.get_cost_updates(y, lr, reg, optimizer[options["optimizer"]])
	
	train_model = theano.function(inputs = [x, y, lr, reg], outputs = cost, updates = updates)
	
	train_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	test_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	
	idx = numpy.arange(train_set_size)
	train_num = 0
	best_err = 1.0
	error_output = open("mlp.txt", "wb")
	with open("model_mlp.npz", "wb") as fout:
		for epoch in range(options["n_epoch"]):
			numpy.random.shuffle(idx)
			new_train_x = [train_x[i] for i in idx]
			new_train_y = [train_y[i] for i in idx]
			for n_batch_index in range(n_train_batch):
				c = train_model(
					new_train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					new_train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					0.01, 0.01
				)
				train_num = train_num + 1
				if train_num%options["print_freq"]==0:
					print("train num: %d, cost: %lf"%(train_num, c))
				
				if train_num%options["valid_freq"]==0:
					train_errors = [train_err(train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 0.01, 0.01) for n_batch_index in range(n_train_batch)]

					test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.01, 0.01) for n_test_index in range(n_test_batch)]
					
					if numpy.mean(test_errors) < best_err:
						best_err = numpy.mean(test_errors)
						
						
						params = dict([(p.name, p.get_value()) for p in model.params])
						numpy.savez(fout, params)
						
						print("train num: %d, best train error: %lf, best test error: %lf"%(train_num, numpy.mean(train_errors), numpy.mean(test_errors)))
			print("epoch %d end"%epoch)
			test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.01, 0.0) for n_test_index in range(n_test_batch)]
			print("%lf"%numpy.mean(test_errors), file=error_output)
	
	
	
	