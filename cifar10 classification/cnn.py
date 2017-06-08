# coding: utf-8
#
# mlp.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of Convolutional Neural Network, Much of the code is modified from deeplearning tutorial:
# http://deeplearning.net/tutorial/lenet.html
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function

from basiclib import *
from mlp import *


class LeNetConvPoolLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		self.input = input
		fan_in = numpy.prod(filter_shape[1:])

		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=image_shape
		)
		
		self.conv_output = T.nnet.relu(conv_out+self.b.dimshuffle('x', 0, 'x', 'x'))

		pooled_out = pool.pool_2d(
			input=self.conv_output,
			ds=poolsize,
			ignore_border=True
		)

		self.output = pooled_out

		self.params = [self.W, self.b]

		self.input = input


def train_cnn(n_epochs=300, nkerns=[100, 100, 100], batch_size=500):
	train_x, train_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/data_batch_*")
	test_x, test_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/test_batch")
	
	train_x = train_x / 255.0
	test_x = test_x / 255.0
	
	
	rng = numpy.random.RandomState(23455)
	train_set_size, col = train_x.shape
	test_set_size, _ = test_x.shape
	
	n_train_batch = train_set_size//batch_size
	n_test_batch = test_set_size//batch_size
	
	x = T.matrix('x').astype(theano.config.floatX)
	y = T.ivector('y')
	lr = T.scalar('lr', dtype=theano.config.floatX)
	reg = T.scalar('reg', dtype=theano.config.floatX)
	
	layer0_input = x.reshape((batch_size, 3, 32, 32))
	
	layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )
	
	layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
	
	layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 6, 6),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )
	
	fc_input = layer2.output.flatten(2)
	
	fc_layer = MLP(fc_input, nkerns[2] * 2 * 2, [1000], options['n_output'])

	params = fc_layer.params + layer2.params + layer1.params + layer0.params
	
	cost = fc_layer.output_layer.cross_entropy(y)
	L = (layer2.W**2).sum() + (layer0.W**2).sum() + (layer1.W**2).sum()
	for hidden in fc_layer.hidden_layers:
		L = L + (hidden.W**2).sum()
	L=L+(fc_layer.output_layer.W**2).sum()
	
	cost = cost + reg*L
	updates = optimizer[options["optimizer"]](cost, params, lr)

	train_model = theano.function(inputs=[x, y, lr, reg], outputs=cost, updates=updates)
	
	train_err = theano.function(inputs = [x, y, lr, reg], outputs = fc_layer.error_rate(y), on_unused_input = 'ignore')
	#valid_err = theano.function(inputs = [x, y, lr, reg], outputs = fc_layer.error_rate(y), on_unused_input = 'ignore')
	test_err = theano.function(inputs = [x, y, lr, reg], outputs = fc_layer.error_rate(y), on_unused_input = 'ignore')
	

	idx = numpy.arange(train_set_size)
	train_num = 0
	best_err = 1.0
	error_output = open("cnn2.txt", "wb")
	with open("model_cnn.npz", "wb") as fout:
		for epoch in range(n_epochs):
			numpy.random.shuffle(idx)
			new_train_x = [train_x[i] for i in idx]
			new_train_y = [train_y[i] for i in idx]
			for n_batch_index in range(n_train_batch):
				c = train_model(
					new_train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					new_train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					0.01, 0.0
				)
				train_num = train_num + 1
				if train_num%options["print_freq"]==0:
					print("train num: %d, cost: %lf"%(train_num, c))
				
				if train_num%options["valid_freq"]==0:
					train_errors = [train_err(train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 0.01, 0.0) for n_batch_index in range(n_train_batch)]
					
					test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.01, 0.0) for n_test_index in range(n_test_batch)]
					
					if numpy.mean(test_errors) < best_err:
						best_err = numpy.mean(test_errors)
						
						
						pp = dict([(p.name, p.get_value()) for p in params])
						numpy.savez(fout, pp)
						
						print("train num: %d, best train error: %lf, best test error: %lf"%(train_num, numpy.mean(train_errors), numpy.mean(test_errors)))
			print("epoch %d end"%epoch)
			test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.01, 0.0) for n_test_index in range(n_test_batch)]
			print("%lf"%numpy.mean(test_errors), file=error_output)
		
if __name__ == '__main__':
	train_cnn()
