
from basiclib import *
		
class HiddenLayer(object):
	def __init__ (self, rng, input, n_input, n_output, activation=T.nnet.relu, W=None, b=None):
		self.input = input
		self.n_input = n_input
		self.n_output = n_output
		
		if W is None: 
			if activation==T.nnet.relu:
				W = (0.01*rng.standard_normal(size=(n_input, n_output))).astype(theano.config.floatX)
			else:
				W = rng.uniform(
					low = -numpy.sqrt(6.0/(n_input+n_output)),
					high = numpy.sqrt(6.0/(n_input+n_output)),
					size = (n_input, n_output)
				).astype(theano.config.floatX)
				if activation==T.nnet.sigmoid:
					W *= 4.0
			W = theano.shared(value=W, name='Hidden_W')
		self.W = W
		
		if b is None:
			b = numpy.zeros(shape=(n_output, )).astype(theano.config.floatX)
			b = theano.shared(value=b, name='Hidden_b')
		self.b = b
	
		self.params = [self.W, self.b]

		self.output = activation(T.dot(input, self.W) + self.b)

class LogisticRegression(object):
	def __init__(self, input, n_input, n_output, W=None, b=None):
		self.input = input
		self.n_input = n_input
		self.n_output = n_output
		if W is None:
			W = numpy.zeros(shape=(n_input, n_output)).astype(theano.config.floatX)
			W = theano.shared(value=W, name='Logistic_W')
		self.W = W
		
		if b is None:
			b = numpy.zeros(shape=(n_output, )).astype(theano.config.floatX)
			b = theano.shared(value=b, name='Logistic_b')
		self.b = b
		
		self.params=[self.W, self.b]
		
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
		self.p_pred = T.argmax(self.p_y_given_x, axis=1)
		
	def cross_entropy(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		
	def errors(self, y):
		return T.mean(T.neq(self.p_pred, y))
		
def _dropout_from_layer(rng, layer, p):
	srng = RandomStreams(rng.randint(12345678))
	return (layer * srng.binomial(n=1, p=p)).astype(theano.config.floatX)

class DropoutHiddenLayer(HiddenLayer):
	def __init__ (self, rng, input, n_input, n_output, dropout_rate=0.8, activation=T.nnet.relu, W=None, b=None):
		super(DropoutHiddenLayer, self).__init__(
			rng=rng, input=input, n_input=n_input, n_output=n_output, W=W, b=b, activation=activation
		)
		
		self.output = _dropout_from_layer(rng, self.output, dropout_rate)


		
class MLPDropout(object):
	def __init__ (self, rng, input, n_input, n_hiddens, n_output, dropout_rates):
		layer_sizes = [n_input] + n_hiddens + [n_output]
		self.weight_matrix_size = zip(layer_sizes[:-1], layer_sizes[1:])
		
		self.hidden_layers, self.dropout_hidden_layers = [], []
		
		for idx, (n_in, n_out) in enumerate(self.weight_matrix_size[:-1]):
			if idx == 0:
				next_input, next_dropout_input = input, _dropout_from_layer(rng, input, dropout_rates[idx])
			else:
				next_input, next_dropout_input = self.hidden_layers[-1].output, self.dropout_hidden_layers[-1].output
			
			self.dropout_hidden_layers.append(DropoutHiddenLayer(rng, next_dropout_input, n_in, n_out, dropout_rates[idx+1], T.nnet.relu))
			
			self.hidden_layers.append(HiddenLayer(
				rng, next_input, n_in, n_out, T.nnet.relu, 
				self.dropout_hidden_layers[-1].W * dropout_rates,
				self.dropout_hidden_layers[-1].b
				)
			)
			
		n_in, n_out = self.weight_matrix_size[-1]
		self.dropout_output_layer = LogisticRegression(self.dropout_hidden_layers[-1].output, n_in, n_out)
		
		self.output_layer = LogisticRegression(self.hidden_layers[-1].output, n_in, n_out,
			self.dropout_output_layer.W*dropout_rates, self.dropout_output_layer.b
		)
		
		
		self.cross_entropy = self.output_layer.cross_entropy
		self.errors = self.output_layer.errors
		
		self.dropout_cross_entropy = self.dropout_output_layer.cross_entropy
		self.dropout_errors = self.dropout_output_layer.errors
		
		self.params = [param for layer in self.dropout_hidden_layers for param in layer.params]
		self.params.extend(self.dropout_output_layer.params)
		
class LeNetConvPoolLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		self.input = input
		fan_in = numpy.prod(filter_shape[1:])

		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			value=numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX),
			name='conv_w', borrow=True
		)

		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, name='conv_b', borrow=True)

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


		#self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.output = pooled_out

		self.params = [self.W, self.b]

		self.input = input
