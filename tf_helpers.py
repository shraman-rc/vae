""" tf_helpers.py: Random relevant TensorFlow augmentations
"""

__author__ = "shraman-rc"

import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

X_INIT=tf.contrib.layers.xavier_initializer(uniform=True, seed=0)

def xavier(dims):
    return X_INIT(dims)

def gaussian(dims):
    return tf.random_normal(dims)

def shape(t):
    return t.get_shape().as_list()
