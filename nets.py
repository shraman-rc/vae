import tensorflow as tf
import tf_helpers as tfh
from collections import namedtuple

BernoulliParam = namedtuple('BernoulliParam', ['p'])
GaussianParam = namedtuple('GaussianParam', ['mu', 'sigma'])

class MLP(object):

    '''
    Wrapper for creating fully-connected Multi-Layer Perceptron models
        that parameterize certain distributions
    '''

    def __init__(self, input_batch, layer_sizes):
        '''
        Params:
            - input_batch: The input tensor to the MLP
            - layer_sizes: The number of weights in each hidden layer
        '''
        self.batch_size, self.input_size = tfh.shape(input_batch)
        self.layer_sizes = layer_sizes
        # Generate hidden layer vars
        self.weights = []
        self.weights.append(tf.Variable(tf.zeros([self.input_size, layer_sizes[0]])))
        for in_size,out_size in zip(layer_sizes, layer_sizes[1:]):
            # TODO: Use tf.contrib.layers.xavier_initializer(...)
            self.weights.append(tf.Variable(tf.zeros([in_size, out_size])))

        # Generate output vars (parameters of a distribution)
        self.params = self._gen_params()

class BernoulliMLP(MLP):

    def __init__(self, input_batch, layer_sizes=[100]):
        super(self.__class__, self).__init__(input_batch, layer_sizes)

    def _gen_params(self, ):

