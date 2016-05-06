""" likelihoods.py: TF implementations of closed-form likelihoods given data/parameters
"""

__author__ = "shraman-rc"

import tensorflow as tf
from collections import namedtuple

'''
Notation:
    - 'll' stands for 'log likelihood'
'''

BARRIER = 1e-8 # Prevents evaluation of log(0) = nan

def ll_bernoulli(data, rho):
    '''
    Params:
        - data: The data point(s) (possibly a batch) for which we are computing
            the likelihood value
        - rho: The probabilities that parameterize a multivariate Bernoulli
    '''
    # Breakdown of the Bernoulli log-likelihood equation:
    # 
    # The data vectors (should) come as binary: {0,1}^n
    # The probability of producing element 'i' in that vector is:
    #   rho_i if x_i = 1, and (1-rho_i) if x_i = 0
    # A concise way to write this probability is:
    #   p(x_i) = x_i * rho_i + (1-x_i)*(1-rho_i)
    # Therefore the likelihood of the entire vector is:
    #   \prod_i {x_i * rho_i + (1-x_i)*(1-rho_i)}
    # Taking the log of this (only the probabilities rho_i, not the
    # 0-1 coefficients x_i) we get our log-likelihood:
    #   \sum_i {x_i*log(rho_i) + (1-x_i)*log(1-rho_i)}
    return tf.reduce_sum(data*tf.log(BARRIER + rho) + (1-data)*tf.log(BARRIER + 1-rho), 1)

def ll_gaussian(data, mu, log_var):
    '''
    Params:
        - data: The data point(s) (possibly a batch) for which we are computing
            the likelihood value
        - mu: Mean of the Gaussian
        - log_var: Log(variance of the Gaussian)
    '''
    # Simply implements the log Normal equation in TF
    return None #TODO
