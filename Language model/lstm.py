
from __future__ import print_function
from basiclib import *


def adadelta(lr, tparams, grads, x, mask, y, cost):
	zipped_grads = [theano.shared(p.get_value()*0.0, name='%s_grad'%k) for k, p in tparams.iteritems()]
	
	running_up2 = [theano.shared(p.get_value()*0.0, name='%s_rup2'%k) for k, p in tparams.iteritems()]
	
	running_grads2 = [theano.shared(p.get_value()*0.0, name='%s_rgrad2'%k) for k, p in tparams.iteritems()]
	
	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95*rg2+0.05*(g**2)) for rg2, g in zip(running_grads2, grads)]
	
	f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	
	dir = [-(T.sqrt(rg+0.00001)*g)/T.sqrt(rg2+0.00001)for rg, g, rg2 in zip(running_up2, zipped_grads, running_grads2)]
	
	params_updates = [(p, p+d) for p, d in zip(tparams.values(), dir)]
	
	ru2up = [(p, 0.95*p+0.05*(d**2)) for p, d in zip(running_up2, dir)]
	
	f_update = theano.function([lr], [], updates= ru2up + params_updates, on_unused_input='ignore', name='adadelta_f_update')
	
	return f_grad_shared, f_update
	
	
class LSTM:
	def __init__ (self, options):
		self.n_words = options['n_words']
		self.n_emb = options['n_emb']
		self.n_hidden = options['n_hidden']
		self.n_output = options['n_output']
		
		self.params = self.init_param()
		self.tparams = self.init_tparam()
		
		x = T.matrix('x', dtype='int64')
		mask = T.matrix('mask', dtype=theano.config.floatX)
		y = T.matrix('y', dtype='int64')
		
		n_steps, n_samples = x.shape 
		xx = self.tparams['emb'][x.flatten()].reshape(shape=(n_steps, n_samples, self.n_emb))
		
		def _slice(data, n, dim):
			return data[:, n*dim:(n+1)*dim]
			
		def _step(x_t, mask, y_t, c_t_prev, s_t_prev, lstm_u, lstm_w, lstm_b, u, b):
			# (n_samples, 4*n_hidden)
			prea = T.dot(x_t, lstm_u) + T.dot(s_t_prev, lstm_w) + lstm_b
			i = T.nnet.sigmoid(_slice(prea, 0, self.n_hidden))
			f = T.nnet.sigmoid(_slice(prea, 1, self.n_hidden))
			c_pi = T.tanh(_slice(prea, 2, self.n_hidden))
			o = T.nnet.sigmoid(_slice(prea, 3, self.n_hidden))
			
			# (n_samples, n_hidden)
			c_t = f*c_t_prev + i*c_pi 
			c_t = mask[:, None]*c_t + (1-mask)[:, None]*c_t_prev
			
			s_t = o*T.tanh(c_t)
			s_t = mask[:, None]*s_t + (1-mask)[:, None]*s_t_prev
			
			# (n_samples, n_output)
			o_t = T.nnet.softmax(T.dot(s_t, u) + b)
			#o_t = mask[:, None]*o_t+T.concatenate(((1-mask)[:,None], T.zeros_like(o_t)[:, 1:]), axis=1)
			
			cost = -T.mean((T.log(o_t)[T.arange(y_t.shape[0]), y_t]) * mask)
			
			return c_t, s_t, o_t, cost
			
		ret, updates = theano.scan(
			fn = _step,
			sequences = [xx, mask, y],
			outputs_info = [T.alloc(0.0, n_samples, self.n_hidden).astype(theano.config.floatX), 
							T.alloc(0.0, n_samples, self.n_hidden).astype(theano.config.floatX),
							None, None],
			non_sequences = [self.tparams['lstm_u'], self.tparams['lstm_w'], 
					self.tparams['lstm_b'], self.tparams['u'], self.tparams['b']],
		)
		
		#p_out = ret[2].reshape((self.n_steps, self.n_samples, self.n_output))
		self.output = ret[2]
		#print(type(self.output))
		self.pred = T.argmax(self.output, axis=2)
		
		f_prob = theano.function(inputs=[x, mask, y], outputs=self.output)
		f_pred = theano.function(inputs=[x, mask, y], outputs=self.pred)
		
		'''
		def _cost_step(out, y):
			#print(y)
			return -T.mean(T.log(out)[T.arange(y.shape[0]), y])
		
		cost1, updates = theano.scan(
			fn=_cost_step,
			sequences = [ret[2], y],
			outputs_info = [None]
		)
		'''
		cost = T.mean(ret[3])
		
		lr = T.scalar('lr', dtype=theano.config.floatX)
		
		grads = T.grad(cost, self.tparams.values())
		
		#self.f_grad_shared, self.f_update = adadelta(lr, self.tparams, grads, x, mask, y, cost)
		#self.f_cost = theano.function(
		#	inputs=[x, mask, y], outputs=cost1 
		#)
		updates = [(p, p-lr*g) for p, g in zip(self.tparams.values(), grads)]
		
		self.train = theano.function(
			inputs = [x, mask, y, lr],
			outputs = cost,
			updates = updates
		)
		
		self.f_grad_shared, self.f_update = adadelta(lr, self.tparams, grads, x, mask, y, cost)


		
	def init_param(self):
		params = OrderedDict()
		params['emb'] = numpy.random.uniform(
				low = -numpy.sqrt(6.0/(self.n_words+self.n_emb)),
				high = numpy.sqrt(6.0/(self.n_words+self.n_emb)),
				size = (self.n_words, self.n_emb)
			).astype(theano.config.floatX)
		
		params['lstm_u'] = numpy.random.uniform(
				low = -numpy.sqrt(6.0/(self.n_emb+self.n_hidden)),
				high = numpy.sqrt(6.0/(self.n_emb+self.n_hidden)),
				size = (self.n_emb, 4*self.n_hidden)
			).astype(theano.config.floatX)
			
		params['lstm_w'] = numpy.random.uniform(
				low = -numpy.sqrt(6.0/(self.n_hidden+self.n_hidden)),
				high = numpy.sqrt(6.0/(self.n_hidden+self.n_hidden)),
				size = (self.n_hidden, 4*self.n_hidden)
			).astype(theano.config.floatX)
			
		params['lstm_b'] = numpy.zeros((4*self.n_hidden, )).astype(theano.config.floatX)
		
		params['u'] = numpy.random.uniform(
				low = -numpy.sqrt(6.0/(self.n_hidden+self.n_output)),
				high = numpy.sqrt(6.0/(self.n_hidden+self.n_output)),
				size = (self.n_hidden, self.n_output)
			).astype(theano.config.floatX)
			
		params['b'] = numpy.zeros((self.n_output, )).astype(theano.config.floatX)
		
		return params
		
	def init_tparam(self):
		tparams = OrderedDict()
		
		for k, v in self.params.iteritems():
			tparams[k] = theano.shared(value=v, name=k)
		
		return tparams


	
def convert(x, y):
	n_samples = len(x)
	lengths = numpy.array([len(e) for e in x])
	max_length = numpy.max(lengths)
	xx = numpy.zeros(shape=(max_length, n_samples)).astype('int64')
	mask = numpy.zeros(shape=(max_length, n_samples)).astype(theano.config.floatX)
	yy = numpy.zeros(shape=(max_length, n_samples)).astype('int64')
	
	for id, (ex, ey) in enumerate(zip(x,y)):
		xx[:lengths[id], id] = ex  
		mask[:lengths[id], id] = numpy.ones((lengths[id], ))
		yy[:lengths[id], id] = ey  
		
	return xx, mask, yy
	
if __name__ == "__main__":
	
	stop_words = set(stopwords.words('english'))
	with open("stop_words.txt", "rb") as fin:
		for line in fin:
			stop_words.add(line.strip())
	#print(stop_words)
	print("load dataset start, time: %s" % (time.strftime("%Y-%m-%d, %X", time.localtime())))		
	with open("large_dataset.csv", "rb") as fin:
		reader = csv.reader(fin)
		reader.next()
		# sentence tokenize
		sentences = itertools.chain(*[(nltk.sent_tokenize(line[0].strip().lower().decode('utf-8'))) for line in reader])
	print("load dataset end, time: %s\n" % (time.strftime("%Y-%m-%d, %X", time.localtime())))
	
	
	print("tokenize start, time: %s" % (time.strftime("%Y-%m-%d, %X", time.localtime())))	
	# word tokenize
	words = [nltk.word_tokenize(sentence) for sentence in sentences]
	#print(words[0])
	#print(len(words))
	
	# remove stop words
	words = [[word for word in sent if (word not in stop_words) and len(word)>1] for sent in words]
	words = filter(lambda x:len(x)>5 and len(x)<1000, words)
	fq = nltk.FreqDist(itertools.chain(*words))
	top_words = fq.most_common(options['n_words']-2)
	#print(fq.B(), fq.N())
	#print(top_words[0:50])
	
	#fq.plot(50, cumulative=True)
	print("tokenize end, time: %s\n" % (time.strftime("%Y-%m-%d, %X", time.localtime())))
	
	
	# create one-hot representation
	word_2_idx, idx_2_word = {}, {}
	for idx, (word, cnt) in enumerate(top_words):
		word_2_idx[word] = idx+1 
		idx_2_word[idx+1] = word
	word_2_idx["unknown_token"] = options['n_words']-1
	idx_2_word[options['n_words']-1] = "unknown_token"
	
	words = numpy.array([[word_2_idx[word] if word in word_2_idx else word_2_idx["unknown_token"] for word in sent] for sent in words])
	#rng = numpy.random.RandomState(100)
	print("build model start, time: %s\n" % (time.strftime("%Y-%m-%d, %X", time.localtime())))
	model = LSTM(options)
	print("build model end, time: %s\n" % (time.strftime("%Y-%m-%d, %X", time.localtime())))
	
	idx = numpy.arange(len(words))
	batch_size = options['batch_size']
	if len(words)%batch_size==0:
		n_batch = len(words)/batch_size
	else:
		n_batch = (len(words)/batch_size)+1
	print("batch_size: %d, n_batch: %d" % (batch_size, n_batch))
	with open("cost_out.txt", "wb") as fout:
		for ep in range(100):
			numpy.random.shuffle(idx)
			avg_cost = []
			for id in range(n_batch):
				batch=idx[id*batch_size:(id+1)*batch_size]
				x = [words[i][:-1] for i in range(5)]
				y = [words[i][1:] for i in range(5)]
				xx, mask, yy = convert(x, y)
				cost = model.f_grad_shared(xx, mask, yy)
				model.f_update(0.01)
				avg_cost.append(cost)
				
			print("epoch: %d, average cost: %lf"%(ep, numpy.mean(avg_cost)))
			print("%lf"%(numpy.mean(avg_cost)), file=fout)

	