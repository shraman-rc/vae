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

initialize = tfh.xavier_init
#initialize = tf.zeros
#activate = tf.nn.softplus
#activate = tf.tanh
activate = tf.nn.relu

class MLP(object):

    '''
    Wrapper for creating fully-connected Multi-Layer Perceptron models
        that parameterize certain distributions
    '''

    def __init__(self, input_batch, layer_sizes, latent_dim, fn_activate=tf.nn.relu,
                    fn_init_w=tfh.xavier_init, fn_init_b=tf.zeros):
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
        # Generate hidden layer vars
        self.weights = [tf.Variable(
                            self.w_init([self.input_dim, layer_sizes[0]])
                        )]
        self.biases = [tf.Variable(
                            self.b_init([layer_sizes[0]])
                      )]
        #for in_dim,out_dim in zip(layer_sizes, layer_sizes[1:]):
        #    # TODO: Use tf.contrib.layers.xavier_initializer(...)
        #    self.weights.append(tf.Variable(initialize([in_dim, out_dim])))
        #    self.biases.append(tf.Variable(tf.zeros([out_dim])))

        # Generate output vars (parameters of a distribution)
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
        # TODO: Extend this to be multilayer
        hidden_weights, hidden_bias, hidden_size = \
            self.weights[0], self.biases[0], self.layer_sizes[0]
        self.bias_out = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_out = tf.Variable(initialize([hidden_size, self.out_dim]))

        # TODO: Tru ReLU instead of tanh
        p = tf.nn.sigmoid(
            tf.matmul(
                activate(tf.matmul(self.input_batch, hidden_weights) + hidden_bias),
                self.weights_out)
            + self.bias_out)

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
        # TODO: Extend this to be multilayer
        hidden_weights, hidden_bias, hidden_size = \
            self.weights[0], self.biases[0], self.layer_sizes[0]
        self.bias_mu = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_mu = tf.Variable(initialize([hidden_size, self.out_dim]))
        self.bias_logvar = tf.Variable(tf.zeros([self.out_dim]))
        self.weights_logvar = tf.Variable(initialize([hidden_size, self.out_dim]))

        # TODO: Tru ReLU instead of tanh
        h = activate(
            tf.matmul(self.input_batch, hidden_weights) + hidden_bias)
        mu = tf.matmul(h, self.weights_mu) + self.bias_mu
        log_var = tf.matmul(h, self.weights_logvar) + self.bias_logvar
            # TODO: Why do we use log variance? Do we expect
            # extremely high/low values? Does it help with
            # vanishing weights phenomenon?

        return GaussianParam(mu,log_var)
