# coding: utf-8
#
# optimization.py
#
# Author: Huang Anbu
# Date: 2017.2
#
# Implement several kinds of commonly used optimization algorithm with theano, including:
# sgd, momentum, nesterov_momentum, adagrad, adadelta, adam, rsprop


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
'''      
def nesterov_momentum(cost, params, velocitys, lr, mu):
    gparams = T.grad(cost, params)
    updates = []
    replaces = dict([(p, p-mu*v) for p, v in itertools.izip(params, velocitys)])
    for param, gparam, velocity in itertools.izip(params, gparams, velocitys):
        #p = param - mu*velocity
        gp = theano.clone(gparam, replace=replaces)
        v = mu*velocity + lr*gp
        updates.append((velocity, v))
        updates.append((param, param-v))
    return updates 
'''     
 
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
        
def rmsprop(cost, params, accs, lr, rho):
	gparams = T.grad(cost, params)
	updates = []
	epsilon = 0.1
	for param, gparam, acc in itertools.izip(params, gparams, accs):
		acc_new = rho*acc + (1 - rho)*(gparam**2)
		updates.append((acc, acc_new))
		
		update = (lr*gparam) / T.sqrt(acc_new+epsilon)        
		
		updates.append((param, param - update))
		
	return updates
    
    
def adam(cost, params, m_t, v_t, beta1, beta2, lr, t):
	gparams = T.grad(cost, params)
	updates = []
	epsilon = 0.1
	for param, gparam in itertools.izip(params, gparams):
		m_t_new = beta1*m_t + (1-beta1)*(gparam)
		v_t_new = beta2*m_t + (1-beta2)*(gparam**2)
		
		step = ((lr*m_t_new)*T.sqrt(1-beta2**t)) / (T.sqrt(1-beta1**t)+epsilon)
		updates.append((m_t, m_t_new))
		updates.append((v_t, v_t_new))
		updates.append((param, param-step))
		
	return updates