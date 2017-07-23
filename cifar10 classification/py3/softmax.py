# coding: utf-8
#
# softmax.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of softmax classification
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function, division

from basiclib import *



# 模型构建
class SoftmaxLayer:
	def __init__ (self, input, n_input, n_output):
		self.input = input
		self.n_input = n_input
		self.n_output = n_output 
		
		# 权重参数初始化
		self.W = theano.shared(
			value = numpy.zeros(shape=(n_input, n_output)).astype(theano.config.floatX), name = "W", borrow = True 
		)
		
		self.b = theano.shared(
			value = numpy.zeros(shape=(n_output, )).astype(theano.config.floatX), name = 'b', borrow = True 
		)
		
		self.params = [self.W, self.b]
		
		# 输出矩阵
		self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W)+self.b)
		
		# 预测值
		self.p_pred = T.argmax(self.p_y_given_x, axis=1)
		
	def cross_entropy(self, y):
		# 交叉熵损失函数
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		
	def get_cost_updates(self, y, lr, reg, optimizer_fun):
		cost = self.cross_entropy(y) + 0.5*reg*((self.W**2).sum())
		try:
			updates = optimizer_fun(cost, self.params, lr)
		except:
			print("Error: no optimizer function")
		else:
			return (cost, updates)
			
	def error_rate(self, y):
		# 错误率
		return T.mean(T.neq(self.p_pred, y))

if __name__ == "__main__":
	# 读取输入数据
	train_x, train_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/data_batch_*")
	test_x, test_y = load_cifar10_dataset(r"./dataset/cifar-10-batches-py/test_batch")
	
	train_x = train_x / 255.0
	test_x = test_x / 255.0
	
	train_set_size, col = train_x.shape
	test_set_size, _ = test_x.shape

	# 设置tensor变量
	x = T.matrix('x').astype(theano.config.floatX)
	y = T.ivector('y')
	index = T.iscalar('index')
	lr = T.scalar('lr', dtype=theano.config.floatX)
	reg = T.scalar('reg', dtype=theano.config.floatX)
	
	batch_size = options['batch_size']
	n_train_batch = train_set_size//batch_size
	n_test_batch = test_set_size//batch_size
	
	
	model = SoftmaxLayer(x, col, options['n_output'])
	cost, updates = model.get_cost_updates(y, lr, reg, optimizer[options["optimizer"]])
	
	# 构建训练函数
	train_model = theano.function(inputs = [x, y, lr, reg], outputs = cost, updates = updates)
	
	# 构建测试函数
	train_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	test_err = theano.function(inputs = [x, y, lr, reg], outputs = model.error_rate(y), on_unused_input = 'ignore')
	
	idx = numpy.arange(train_set_size)
	train_num = 0
	best_err = 1.0
	error_output = open("softmax.txt", "w")
	with open("model_softmax.npz", "wb") as fout:
		for epoch in range(options["n_epoch"]):
			numpy.random.shuffle(idx)
			new_train_x = [train_x[i] for i in idx]
			new_train_y = [train_y[i] for i in idx]
			for n_batch_index in range(n_train_batch):
				c = train_model(
					new_train_x[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					new_train_y[n_batch_index*batch_size:(n_batch_index+1)*batch_size], 
					0.0001, 0.0
				)
				train_num = train_num + 1
				if train_num%options["print_freq"]==0:
					print("train num: %d, cost: %lf"%(train_num, c))
				
				if train_num%options["valid_freq"]==0:
					train_errors = [train_err(train_x[n_train_index*batch_size:(n_train_index+1)*batch_size], train_y[n_train_index*batch_size:(n_train_index+1)*batch_size], 0.00000001, 0.0) for n_train_index in range(n_train_batch)]

					test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.00000001, 0.0) for n_test_index in range(n_test_batch)]
					
					if numpy.mean(test_errors) < best_err:
						best_err = numpy.mean(test_errors)
						
						
						params = dict([(p.name, p.get_value()) for p in model.params])
						numpy.savez(fout, params)
						
						print("train num: %d, best train error: %lf, best test error: %lf"%(train_num, numpy.mean(train_errors), numpy.mean(test_errors)))
			print("epoch %d end" % epoch)
			test_errors = [test_err(test_x[n_test_index*batch_size:(n_test_index+1)*batch_size], test_y[n_test_index*batch_size:(n_test_index+1)*batch_size], 0.00000001, 0.0) for n_test_index in range(n_test_batch)]
			print("%lf" % numpy.mean(test_errors), file=error_output)