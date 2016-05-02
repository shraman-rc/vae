import tensorflow as tf
import tf_helpers as tfh
from collections import namedtuple


DEFAULT_LAYER_SIZES = [100]
DEFAULT_LATENT_DIM = 10

BernoulliParam = namedtuple('BernoulliParam', ['p'])
GaussianParam = namedtuple('GaussianParam', ['mu', 'sigma'])

class MLP(object):

    '''
    Wrapper for creating fully-connected Multi-Layer Perceptron models
        that parameterize certain distributions
    '''

    def __init__(self, input_batch, layer_sizes, dim):
        '''
        Params:
            - input_batch: The input tensor to the MLP
            - layer_sizes: The number of weights in each hidden layer
            - dim: dimension of the resulting probability distribution
        '''
        self.batch_size, self.input_size = tfh.shape(input_batch)
        self.layer_sizes = layer_sizes
        self.out_dim = dim
        # Generate hidden layer vars
        self.weights = [tf.Variable(tf.zeros([self.input_size, layer_sizes[0]]))]
        self.biases = [tf.Variable(tf.zeros([layer_sizes[0]]))]
        for in_size,out_size in zip(layer_sizes, layer_sizes[1:]):
            # TODO: Use tf.contrib.layers.xavier_initializer(...)
            self.weights.append(tf.Variable(tf.zeros([in_size, out_size])))
            self.biases.append(tf.Variable(tf.zeros([out_size])))

        # Generate output vars (parameters of a distribution)
        self.params = self._gen_params()

class BernoulliMLP(MLP):

    def __init__(self, input_batch,
            layer_sizes=DEFAULT_LAYER_SIZES, dim=DEFAULT_LATENT_DIM):
        super(self.__class__, self).__init__(input_batch, layer_sizes, dim)

    def _gen_params(self):
        '''
        Setup TF computation graph to compute the output value that
            parameterizes the Bernoulli distribution
        '''
        # TODO: Extend this to be multilayer
        param = BernoulliParam()
        bias_p = tf.Variable(tf.zeros([self.out_dim]))
        weights_p = tf.Variable(tf.zeros([self.layer_sizes[-1], self.out_dim]))

        param.p = tf.sigmoid(tf.matmul( , ) + bias_p)

        return param
