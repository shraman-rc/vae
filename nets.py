""" nets.py: Defines and recontructs basic neural net architectures for the auto-encoder

Construction closely follows [1]: Appendix C
"""

__author__ = "shraman-rc"

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

    def __init__(self, input_batch, layer_sizes, latent_dim, fn_activate=tf.nn.relu,
                    fn_init_w=tfh.xavier, fn_init_b=tf.zeros):
        '''
        Params:
            - input_batch: The input tensor to the MLP
                - NOTE: This should be a 'tf.placeholder'
            - layer_sizes: The number of weights in each hidden layer
            - latent_dim: dimension of the resulting probability distribution
            - fn_activate: function to use for perceptron activation
            - fn_init_w: function to use for weight initialization
            - fn_init_b: function to use for bias initialization
        '''
        self.batch_size, self.input_dim = tfh.shape(input_batch)
        self.input_batch = input_batch
        self.layer_sizes = layer_sizes
        self.out_dim = latent_dim
        self.activate, self.w_init, self.b_init = fn_activate, fn_init_w, fn_init_b

        # Generate input layer
        self.weights = [tf.Variable(
            self.w_init([self.input_dim, layer_sizes[0]]))]
        self.biases = [tf.Variable(
            self.b_init([layer_sizes[0]]))]

        # Keeps track of the hidden layer output as we build
        self.hidden_out = self.activate(
            tf.matmul(self.input_batch, self.weights[-1]) + self.biases[-1])

        # Generate arbitrarily deep hidden layers
        for in_dim,out_dim in zip(layer_sizes, layer_sizes[1:]):
            self.weights.append(tf.Variable(self.w_init([in_dim, out_dim])))
            self.biases.append(tf.Variable(self.b_init([out_dim])))
            self.hidden_out = self.activate(
                tf.matmul(self.hidden_out, self.weights[-1]) + self.biases[-1])

        # Generate output layer (parameters of a distribution)
        self.out_params = self._gen_params()


class BernoulliMLP(MLP):

    def __init__(self, input_batch,
            layer_sizes=DEFAULT_LAYER_SIZES, dim=DEFAULT_LATENT_DIM):
        super(self.__class__, self).__init__(input_batch, layer_sizes, dim)

    def _gen_params(self):
        '''
        Setup TF computation graph (neural net) to compute the value that
            parameterizes the Bernoulli distribution
        '''
        self.bias_out = tf.Variable(self.b_init([self.out_dim]))
        self.weights_out = tf.Variable(self.w_init([self.layer_sizes[-1], self.out_dim]))

        # The output is a (multivariate) probability vector that represents
        #   the "success probabilities" in a Bernoulli dist.
        p = tf.nn.sigmoid(tf.matmul(self.hidden_out, self.weights_out) + self.bias_out)

        return BernoulliParam(p)


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
        self.bias_mu = tf.Variable(self.b_init([self.out_dim]))
        self.weights_mu = tf.Variable(self.w_init([self.layer_sizes[-1], self.out_dim]))

        self.bias_logvar = tf.Variable(self.b_init([self.out_dim]))
        self.weights_logvar = tf.Variable(self.w_init([self.layer_sizes[-1], self.out_dim]))

        mu = tf.matmul(self.hidden_out, self.weights_mu) + self.bias_mu
        # TODO: Why do we use log? Vanishing weights phenomenon when training?
        log_var = tf.matmul(self.hidden_out, self.weights_logvar) + self.bias_logvar

        return GaussianParam(mu,log_var)
