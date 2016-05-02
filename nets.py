'''
nets.py: Defines and constructs basic neural net architectures for the auto-encoder

Construction closely follows [1]: Appendix C
'''

import tensorflow as tf
import tf_helpers as tfh
from collections import namedtuple


DEFAULT_LAYER_SIZES = [100]
DEFAULT_LATENT_DIM = 10

BernoulliParam = namedtuple('BernoulliParam', ['p'])
GaussianParam = namedtuple('GaussianParam', ['mu', 'log_var'])


class MLP(object):

    '''
    Wrapper for creating fully-connected Multi-Layer Perceptron models
        that parameterize certain distributions
    '''

    def __init__(self, input_batch, layer_sizes, dim):
        '''
        Params:
            - input_batch: The input tensor to the MLP
                - NOTE: This should be a 'tf.placeholder'
            - layer_sizes: The number of weights in each hidden layer
            - dim: dimension of the resulting probability distribution
        '''
        self.batch_size, self.input_dim = tfh.shape(input_batch)
        self.input_batch = input_batch
        self.layer_sizes = layer_sizes
        self.out_dim = dim
        # Generate hidden layer vars
        self.weights = [tf.Variable(tf.zeros([self.input_dim, layer_sizes[0]]))]
        self.biases = [tf.Variable(tf.zeros([layer_sizes[0]]))]
        for in_dim,out_dim in zip(layer_sizes, layer_sizes[1:]):
            # TODO: Use tf.contrib.layers.xavier_initializer(...)
            self.weights.append(tf.Variable(tf.zeros([in_dim, out_dim])))
            self.biases.append(tf.Variable(tf.zeros([out_dim])))

        # Generate output vars (parameters of a distribution)
        self.params = self._gen_params()


class BernoulliMLP(MLP):

    def __init__(self, input_batch,
            layer_sizes=DEFAULT_LAYER_SIZES, dim=DEFAULT_LATENT_DIM):
        super(self.__class__, self).__init__(input_batch, layer_sizes, dim)

    def _gen_params(self):
        '''
        Setup TF computation graph (neural net) to compute the value that
            parameterizes the Bernoulli distribution
        '''
        param = BernoulliParam()

        # TODO: Extend this to be multilayer
        hidden_weights, hidden_bias, hidden_size = \
            self.weights[0], self.biases[0], self.layer_sizes[0]
        self.bias_out = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_out = tf.Variable(tf.zeros([hidden_size, self.out_dim]))

        # TODO: Tru ReLU instead of tanh
        param.p = tf.sigmoid(
            tf.matmul(
                tf.tanh(tf.matmul(self.input_batch, hidden_weights) + hidden_bias),
                self.weights_out)
            + self.bias_out)

        return param


class GaussianMLP(MLP):

    def __init__(self, input_batch,
            layer_sizes=DEFAULT_LAYER_SIZES, dim=DEFAULT_LATENT_DIM):
        super(self.__class__, self).__init__(input_batch, layer_sizes, dim)

    def _gen_params(self):
        '''
        Setup TF computation graph (neural net) to compute the values that
            parameterize the Gaussian distribution

        NOTE: The variance parameter output is actually log(variance) as
            suggested by [1]
        '''
        param = GaussianParam()

        # TODO: Extend this to be multilayer
        hidden_weights, hidden_bias, hidden_size = \
            self.weights[0], self.biases[0], self.layer_sizes[0]
        self.bias_mu = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_mu = tf.Variable(tf.zeros([hidden_size, self.out_dim]))
        self.bias_logvar = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_logvar = tf.Variable(tf.zeros([hidden_size, self.out_dim]))

        # TODO: Tru ReLU instead of tanh
        h = tf.tanh(
            tf.matmul(self.input_batch, hidden_weights) + hidden_bias)
        param.mu = tf.matmul(h, self.weights_mu) + self.bias_mu
        param.log_var = tf.matmul(h, self.weights_logvar) + self.bias_logvar
            # TODO: Why do we use log variance? Do we expect
            # extremely high/low values? Does it help with
            # vanishing weights phenomenon?

        return param
