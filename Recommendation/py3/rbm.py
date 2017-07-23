# coding: utf-8
#
# rbm.py
#
# Author: Huang Anbu
# Date: 2017.4
#
# Description: rbm-based collaborative filtering algorithm
#
# contrastive divergence, which is proposed by Geoffrey Hinton, is most commonly use algorithm 
#   in training RBM model
#
# Dataset citation:
#   F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
#       and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
#       Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872 
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from __future__ import print_function
from basiclib import *
	
class RBM(object):
	def __init__(
		self,
		input=None,
		n_visible=784,
		n_hidden=500,
		W=None,
		hbias=None,
		vbias=None,
		numpy_rng=None,
		theano_rng=None
	):
		self.input = input
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			# create a number generator
			numpy_rng = numpy.random.RandomState(1234)

		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if W is None:
			initial_W = numpy.asarray(
				numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					size=(n_visible, n_hidden)
				),
				dtype=theano.config.floatX
			)
			# theano shared variables for weights and biases
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if hbias is None:
			hbias = theano.shared(
				value=numpy.zeros(
					n_hidden,
					dtype=theano.config.floatX
				),
				name='hbias',
				borrow=True
			)

		if vbias is None:
			vbias = theano.shared(
				value=numpy.zeros(
					n_visible,
					dtype=theano.config.floatX
				),
				name='vbias',
				borrow=True
			)
			
		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng
		self.params = [self.W, self.hbias, self.vbias]

	def free_energy(self, v_sample):
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term

	def propup(self, vis):
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
		return pre_sigmoid_h1, h1_mean, h1_sample

	def propdown(self, hid):
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	
	def sample_v_given_h(self, mask, h0_sample):
		'''For recommendation, use softmax instead of sigmoid'''
		
		pre_activation = T.dot(h0_sample, self.W.T) + self.vbias  # (n_visible, )
		#sz = pre_activation.shape[0]
		pre_activation = pre_activation.reshape((int(self.n_visible/5), 5))
		state = T.argmax(pre_activation, axis=1)
		output = T.zeros_like(pre_activation).astype(theano.config.floatX)
		ret = T.set_subtensor(output[T.arange(state.shape[0]), state], 1.0).reshape(mask.shape)
		return ret * mask

	def gibbs_hvh(self, h0_sample, mask):
		v1_sample = self.sample_v_given_h(mask, h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample, mask):
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		v1_sample = self.sample_v_given_h(mask, h1_sample)
		return [h1_sample, v1_sample]

	# start-snippet-2
	def get_cost_updates(self, mask, lr=0.1, persistent=None, k=1):

		pre_h_mean, h_mean, ph_sample = self.sample_h_given_v(self.input)

		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		(
			[
				nv_samples,
				pre_nh_mean, 
				nh_mean,
				nh_samples
			],
			updates
		) = theano.scan(
			self.gibbs_hvh,
			outputs_info=[None, None, None, chain_start],
			n_steps=k,
			non_sequences = [mask],
			name="gibbs_hvh"
		)

		chain_end = nv_samples[-1]

		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

		gparams = T.grad(cost, self.params, consider_constant=[chain_end])
		#gw = T.dot(self.input.reshape((self.n_visible, 1)), h_mean.reshape((1, self.n_hidden))) - T.dot(chain_end.reshape((self.n_visible, 1)), nh_mean[-1].reshape((1, self.n_hidden)))
		#ghbias = h_mean.reshape(self.hbias.shape) - nh_mean[-1].reshape(self.hbias.shape)
		#gvbias = self.input.reshape(self.vbias.shape) - chain_end.reshape(self.vbias.shape)
		
		#gparams = [gw, ghbias, gvbias]
	
		for gparam, param in zip(gparams, self.params):
			# make sure that the learning rate is of the right dtype
			updates[param] = param - gparam * T.cast(
				lr,
				dtype=theano.config.floatX
			)
		if persistent:
			updates[persistent] = nh_samples[-1]

		return cost, updates

	def get_pseudo_likelihood_cost(self, updates):
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')
		xi = T.round(self.input)
		fe_xi = self.free_energy(xi)
		xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
		fe_xi_flip = self.free_energy(xi_flip)

		cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
															fe_xi)))

		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

		return cost

	def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
		cross_entropy = T.mean(
			T.sum(
				self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
				(1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
				axis=1
			)
		)

		return cross_entropy
		
	def get_reconstruction(self, x, mask):
		[h1_sample, v1_sample] = self.gibbs_vhv(x, mask)
		err = []
		x=x.astype('int8')
		v1_sample=v1_sample.astype('int8')
		return 1.0 - T.mean(T.all(T.eq(x, v1_sample).reshape((int(self.n_visible/5), 5)), axis=1))


def train_rbm():
	
	#lr = options["lr"]
	batch_size = options["batch_size"]
	n_hidden = options["n_hidden"]
	
	with open("data.pkl", "rb") as fin:
		min_user_id, max_user_id, min_movie_id, max_movie_id, train_set = pickle.load(fin)
		
	print(min_user_id, max_user_id, min_movie_id, max_movie_id)
	
	HS, WS = train_set.shape
	print(HS, WS)
	new_train_set = numpy.zeros((HS, WS*5))
	new_train_mask = numpy.zeros((HS, WS*5))
	for row in range(HS):
		for col in range(WS):
			r = int(train_set[row][col]) # (user, movie) = r 
			if r==0:
				continue
			new_train_set[row][col*5+r-1] = 1
			new_train_mask[row][col*5:col*5+5] = 1

	print(numpy.mean(new_train_mask))

	new_train_set = new_train_set.astype(theano.config.floatX)
	new_train_mask = new_train_mask.astype(theano.config.floatX)
	
	n_train_batches = new_train_set.shape[0] // batch_size

	x = T.matrix('x')  # the data is presented as rasterized images
	mask = T.matrix('mask')
	cd_k = T.iscalar('cd_k')
	lr = T.scalar('lr', dtype=theano.config.floatX)
	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

	# construct the RBM class
	rbm = RBM(input=x, n_visible=WS*5, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	cost, updates = rbm.get_cost_updates(mask, lr=lr, persistent=persistent_chain, k=cd_k)

	
	train_model = theano.function([x, mask, cd_k, lr], outputs=cost, updates=updates, name='train_rbm')
	
	check_model = theano.function([x, mask], outputs=rbm.get_reconstruction(x, mask), name='check_model')
	numpy.set_printoptions(threshold='nan') 
	
	output = open("output_persistent_k3_lr0.1.txt", "w")
	for epoch in range(20):
		mean_cost = []
		error = []
		p = []	
		for batch_index in range(n_train_batches):
			if epoch<3:
				cd_k = 1
			else:
				cd_k = 2 + (epoch - 3)/2
				
			if epoch<3:
				lr = 0.05
			else:
				lr = 0.05 + ((epoch - 3)/2)*0.01
				
			batch_data = new_train_set[batch_index*batch_size:(batch_index+1)*batch_size]
			batch_data_mask = new_train_mask[batch_index*batch_size:(batch_index+1)*batch_size]
		
			mean_cost += [train_model(batch_data, batch_data_mask, cd_k, lr)]
			
			error += [check_model(batch_data, batch_data_mask)]

		p = []
		
		print("epoch %d end, cost: %lf"%(epoch, numpy.mean(mean_cost)))

		print("epoch %d end, error: %lf"%(epoch, numpy.mean(error)))
		
		print("%lf"%(numpy.mean(error)), file=output)
	
	
if __name__ == '__main__':
	train_rbm()
	#make_recom()