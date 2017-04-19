# coding: utf-8
#
# optimization.py
#
# Author: Huang Anbu
# Date: 2017.3
#
# Description: Implementation of optimization methods, including:
# 		sgd
# 		momentum
# 		nesterov_momentum
# 		adagrad
#		adadelta 
#		rmsprop
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================


from basiclib import *


def sgd(cost, params, lr):
	#lr = T.scalar('lr', dtype=theano.config.floatX)
	gparams = T.grad(cost, params)
	updates = [(param, param - lr*gparam) for param, gparam in zip(params, gparams)]
	return updates 
    
def momentum(cost, params, lr, mu=0.5):
	#lr, mu = T.scalars(2).astype(theano.config.floatX)
	velocity = [theano.shared(value=numpy.zeros_like(p.get_value())) for p in params]
	gparams = T.grad(cost, params)
	updates = []
	for p, g, v in itertools.izip(params, gparams, velocity):
		dir = mu * v + lr * g  
		updates.append((v, dir))
		updates.append((p, p - dir))
	return updates 
 
def nesterov_momentum(cost, params, lr, mu=0.5):
	#params = [param - mu*v for param, v in itertools.izip(params, velocitys)]
	#gparams = [T.grad(cost, param) for param, v in itertools.izip(params, velocitys)]
	#T.grad(cost, params-mu)
	gparams = T.grad(cost, params)
	updates = []
	velocitys = [theano.shared(value=numpy.zeros_like(p.get_value()).astype(theano.config.floatX)) for p in params]
	#replaces = dict([(p, p-mu*v) for p, v in itertools.izip(params, velocitys)])
	for param, gparam, velocity in itertools.izip(params, gparams, velocitys):
		#p = param - mu*velocity
		updates.append((param, param-mu*mu*velocity-lr*(mu+1)*gparam))
		updates.append((velocity, mu*velocity+lr*gparam))
	return updates 

	
def adagrad(cost, params, lr):
	gparams = T.grad(cost, params)
	updates = []
	e = 0.01
	acc = [theano.shared(value=numpy.zeros_like(p.get_value()).astype(theano.config.floatX)) for p in params]
	for param, gparam, acc in itertools.izip(params, gparams, accs):
		updates.append((acc, acc + gparam**2)) 
		updates.append((param, param - (lr/(T.sqrt(acc+e)))*gparam))
	return updates
    
def adadelta(cost, params, lr, rho=0.95):
	gparams = T.grad(cost, params)
	updates = []
	epsilon = 0.0001
	accs = [theano.shared(value=numpy.zeros_like(p.get_value()).astype(theano.config.floatX)) for p in params]
	delta_accs = [theano.shared(value=numpy.zeros_like(p.get_value()).astype(theano.config.floatX)) for p in params]
	for param, gparam, acc, delta_acc in itertools.izip(params, gparams, accs, delta_accs):
		acc_new = rho*acc + (1 - rho)*(gparam**2)
		updates.append((acc, acc_new))
		
		update = (T.sqrt(delta_acc+epsilon)*gparam) / T.sqrt(acc_new+epsilon)  
		
		updates.append((param, param - lr*update))
		
		delta_acc_new = rho*delta_acc + (1 - rho)*(update**2)
		updates.append((delta_acc, delta_acc_new))
		
	return updates
        
def rmsprop(cost, params, lr, rho=0.95):
	gparams = T.grad(cost, params)
	updates = []
	epsilon = 0.0001
	accs = [theano.shared(value=numpy.zeros_like(p.get_value()).astype(theano.config.floatX)) for p in params]
	for param, gparam, acc in itertools.izip(params, gparams, accs):
		acc_new = rho*acc + (1 - rho)*(gparam**2)
		updates.append((acc, acc_new))
		
		update = (lr*gparam) / T.sqrt(acc_new+epsilon)        
		
		updates.append((param, param - update))
		
	return updates
    