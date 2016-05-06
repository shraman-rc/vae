#!/usr/bin/python

""" vae.py: Run the Variational Auto-encoder.
"""

__author__ = "shraman-rc"

import tensorflow as tf
import numpy as np
import click as cl

from nets import BernoulliMLP, GaussianMLP
import likelihoods as lh
import vis

MNIST_FLAT_DIM = 784    # Flattened dimension of MNIST images, 'x'
HIDDEN_LAYER_ENC = 500
HIDDEN_LAYER_DEC = 200


# Experimental Parameters
M = 100         # minibatch size
L = 1           # MCE samples
J = 10          # Dimensionality of latent space, 'z'
T = 5000        # Training iterations
ADG_RATE = 0.1 # Adagrad global learning rate (chosen from
                #   {0.01,0.02,0.1} -- cf. [1]: Section 5)

# Inputs - mini-batches of (flattened) images
x_batch = tf.placeholder(tf.float32, shape=[M, MNIST_FLAT_DIM])

encoder = GaussianMLP(x_batch, [HIDDEN_LAYER_ENC], J) # Gaussian q(z|x)

# Bridge between encoder and decoder
mu_q = encoder.out_params.mu
log_var_q = encoder.out_params.log_var
var_q = tf.exp(log_var_q)
sigma_q = tf.sqrt(var_q)
# Latent space samples w/ reparameterization (z = g(ep,x); ep ~ p(ep))
#   This outsources randomness to auxiliary r.v. (see [1]: Sec 2.4)
#   For a Gaussian 'q' we have: g(ep,x) = mu + sigma*ep, and p(ep) = N(0,I)
# Note: Element-wise univariate Gaussian sampling <=> multivariate Gaussian sampling
# TODO: Introduce L later here
# TODO: Is TF resampling ep everytime?
ep = tf.random_normal([M, J], mean=0, stddev=1)
z_batch = mu_q + sigma_q*ep # one latent per datapoint (MxJ)

decoder = BernoulliMLP(z_batch, [HIDDEN_LAYER_DEC], MNIST_FLAT_DIM)

#
## Weights
#W_hidden_e = tf.Variable(tf.zeros([MNIST_FLAT_DIM, HIDDEN_LAYER_ENC]))
#W_mu_q = tf.Variable(tf.zeros([HIDDEN_LAYER_ENC, J]))
#W_log_sigma_q = tf.Variable(tf.zeros([HIDDEN_LAYER_ENC, J]))
#
## Biases
#b_hidden_e = tf.Variable(tf.zeros([HIDDEN_LAYER_ENC]))
#b_mu_q = tf.Variable(tf.zeros([J]))
#b_log_sigma_q = tf.Variable(tf.zeros([J]))
#
## Activations/Outputs
#h_e = tf.tanh(  # TODO: try ReLU instead of tanh?
#        tf.matmul(x_batch, W_hidden_e) + b_hidden_e)
#mu_q = tf.matmul(h_e, W_mu_q) + b_mu_q
#log_sigma_sq_q = tf.matmul(h_e, W_log_sigma_q) + b_log_sigma_q)
#    # TODO: Why do we use log variance? Do we expect
#    # extremely high/low values? Does it help with
#    # vanishing weights phenomenon?


# Decoder construction (Bernoulli, for MNIST)

## Weights
#W_hidden_d = tf.Variable(tf.zeros([J, HIDDEN_LAYER_DEC]))
#W_mu_p = tf.Variable(tf.zeros([HIDDEN_LAYER_DEC, MNIST_FLAT_DIM]))
#W_log_sigma_p = tf.Variable(tf.zeros([HIDDEN_LAYER_DEC, MNIST_FLAT_DIM]))
#
## Biases
#b_hidden_d = tf.Variable(tf.zeros([HIDDEN_LAYER_DEC]))
#b_mu_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))
#b_log_sigma_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))
#
## Activations/Outputs
#h_d = tf.tanh(  # TODO: try ReLU instead of tanh?
#        tf.matmul(z, W_hidden_d) + b_hidden_d)
#mu_p = tf.matmul(h_d, W_mu_p) + b_mu_p
#log_sigma_sq_p = tf.matmul(h_d, W_log_sigma_p) + b_log_sigma_p)
#    # TODO: Why do we use log variance? Do we expect
#    # extremely high/low values? Does it help with
#    # vanishing weights phenomenon?

# The (negative) KL divergence between the variational approx. and the *prior*
#   p_theta(z) acts as a regularizing term so that the latent distribution
#   doesn't overfit. The closed-form eq. is derived in [1]: Appedix B
# TODO: Why would overfitting be a problem in the auto-encoding scenario? Wouldn't
#   overfitting lead to a better likelihood lower bound measure used in [1]'s experiments?
# TODO: Bug changing mu_q to mu_q**2 makes log likelihood go to zero...
KL_regularizer = 0.5 * tf.reduce_sum(1 + log_var_q - mu_q**2 - var_q, 1)

# The 'reconstruction error' (predictive likelihood): log p_theta(x_batch|z)
# TODO: Try a different loss function? Cross entropy?
reconstr_err = lh.ll_bernoulli(x_batch, decoder.out_params.p)

# TODO: WE TAKE ELEMENT WISE SIGMOID TO PARAMETERIZE MULTIVARIATE BERNOULLI (i.e. pixel values in image - note will need to rescale if doing on regular images) BUT HOW DOES STEEPNESS OF SIGMOID AFFECT LEARNING? Does it take longer for theta (Bernoulli probs) to converge?

# ELBO estimator construction
# TODO: Apply batch normalization here as in [3]?
# TODO: See effects on Adagrad, SGD, and ADAM separately?
ELBO_estimate = tf.reduce_mean(KL_regularizer + reconstr_err) # Mean val over minibatch

# TODO: Using Adagrad as per [1] but was written before ADAM (by same author!)
#       Later transition to ADAM
# Notice that we are minimizing the negative (i.e. maximizing) the variational
#   lower bound
train_op = tf.train.AdagradOptimizer(ADG_RATE).minimize(-ELBO_estimate)

# Train on MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
ELBOs = np.zeros(T)
KLs = np.zeros(T)
LLs = np.zeros(T)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    ELBO, neg_KL, ll = 0,0,np.NINF
    # Train TODO: Move into MLP class
    iters = xrange(T)
    for t in iters:
        cl.secho('Minibatch {}'.format(t), fg='green', bold=False)
        batch = mnist.train.next_batch(M)
        _, ELBO, ll, neg_KL = sess.run(
            [train_op, ELBO_estimate, reconstr_err, KL_regularizer],
            feed_dict={x_batch: batch[0]})
        ELBO, neg_KL, ll = np.mean(ELBO), np.mean(neg_KL), np.mean(ll)
        ELBOs[t] = ELBO; KLs[t] = -neg_KL; LLs[t] = ll; opts[t] = opt
        cl.secho('ELBO (estimate): {}\nKL Div (prior): {}\nLikelihood: {}'
            .format(ELBO, -neg_KL, ll), fg='cyan')

    # Tests
    titles = ["$\mathcal{L}(\phi,\\theta;x)$",
              "$KL(q_{\phi}(z|x)||p_{\\theta}(z))$",
              "$\log(p_{\\theta}(x|z))$"]
    params = {"\eta_{Adagrad}":ADG_RATE, "Activation":"ReLU"}
    vis.basic_multiplot([iters]*3,[[ELBOs], [KLs], [LLs]], titles, show_legend=False, params=params)
    

# TODO: When implementating Variational Dropout, can use tf.nn.dropout
# TODO: Record error vectors

cl.secho('Success!', fg='green', bold=True)
