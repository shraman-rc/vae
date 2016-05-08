""" tf_helpers.py: Random relevant TensorFlow augmentations
"""

__author__ = "shraman-rc"

import tensorflow as tf
import numpy as np

X_INIT=tf.contrib.layers.xavier_initializer(uniform=True, seed=0)

#tf.set_random_seed(0)
#
#def xavier_init(dims, constant=1):
#    """ Xavier initialization of network weights"""
#    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
#    low = -constant*np.sqrt(6.0/(dims[0] + dims[1]))
#    high = constant*np.sqrt(6.0/(dims[0] + dims[1]))
#    return tf.random_uniform((dims[0], dims[1]),
#                             minval=low, maxval=high,
#                             dtype=tf.float32)
def xavier_init(dims):
    return X_INIT(dims)

def shape(t):
    return t.get_shape().as_list()
