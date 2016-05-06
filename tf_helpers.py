""" tf_helpers.py: Random relevant TensorFlow augmentations
"""

__author__ = "shraman-rc"

import tensorflow as tf

def shape(t):
    return t.get_shape().as_list()
