#!/usr/bin/python

""" vae.py: Run the Variational Auto-encoder.
"""

__author__ = "shraman-rc"

import tensorflow as tf
import numpy as np
import click as cl
import yaml

tf.set_random_seed(0)

from nets import BernoulliMLP, GaussianMLP
import likelihoods as lh
import vis

CONFIG = 'config/nn_config.yaml'

try:
    config = yaml.load(file(CONFIG, 'r'))
except yaml.YAMLError, exc:
    cl.secho("Error in configuration file: {}".format(exc), fg='red')


ARCH = config["architecture"]
DIMS = config["dims"]
OPT = config["optimization"]
TRAIN = config["training"]
PARAMS = config["AEVB"]


# Inputs - mini-batches of (flattened) images
x_batch = tf.placeholder(tf.float32, shape=[TRAIN["batch_size"], DIMS["data"]])

# Encoder parameterizes Gaussian approximation q(z|x)
encoder = GaussianMLP(x_batch, ARCH["encoder"]["n_units"], DIMS["latent"])

# Bridge between encoder and decoder
mu_q = encoder.out_params.mu
log_var_q = encoder.out_params.log_var
var_q = tf.exp(log_var_q)
sigma_q = tf.sqrt(var_q)

# Latent space samples w/ reparameterization (z = g(ep,x); ep ~ p(ep))
#   Note: Element-wise univariate Gaussian sampling <=> multivariate Gaussian sampling
# TODO: Introduce L later here
ep = tf.random_normal([TRAIN["batch_size"], DIMS["latent"]], mean=0, stddev=1)
z_batch = mu_q + sigma_q*ep # one latent per datapoint (MxJ)

decoder = BernoulliMLP(z_batch, ARCH["decoder"]["n_units"], DIMS["data"])

# The (negative) KL divergence between the variational approx. and the *prior*
#   p_theta(z) acts as a regularizing term so that the latent distribution
#   doesn't overfit. The closed-form eq. is derived in [1]: Appedix B
# TODO: Why would overfitting be a problem in the auto-encoding scenario? Wouldn't
#   overfitting lead to a better likelihood lower bound measure used in [1]'s experiments?
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
if OPT["type"].lower() == "adagrad":
    optimizer = tf.train.AdagradOptimizer(OPT["Adagrad_rate"])
elif OPT["type"].lower() == "adam":
    optimizer = tf.train.AdamOptimizer(OPT["Adam_rate"])

# Notice that we are minimizing the negative (i.e. maximizing) the variational
#   lower bound
train_op = optimizer.minimize(-ELBO_estimate)

# Train on MNIST
T = TRAIN["n_iters"]
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
ELBOs = np.zeros(T)
KLs = np.zeros(T)
LLs = np.zeros(T)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Train TODO: Move into MLP class
    iters = xrange(T)
    for t in iters:
        cl.secho('Minibatch {}'.format(t), fg='green', bold=False)
        batch = mnist.train.next_batch(TRAIN["batch_size"])
        _, ELBO, ll, neg_KL, mu, log_var, p = sess.run(
            [train_op, ELBO_estimate, reconstr_err, KL_regularizer, mu_q, log_var_q, decoder.out_params.p],
            feed_dict={x_batch: batch[0]})
        ELBO, neg_KL, ll, mu, log_var, p = \
            (np.mean(ELBO),
             np.mean(neg_KL),
             np.mean(ll),
             np.mean(mu,0),
             np.mean(log_var,0),
             np.mean(p,0))
        ELBOs[t] = ELBO; KLs[t] = -neg_KL; LLs[t] = ll
        cl.secho(("ELBO (estimate): {}\n"
        "KL Div (prior): {}\n"
        "Likelihood: {}\n"
        "Mu: {}\n"
        "Log var: {}\n"
        "len(p): {}")
            .format(ELBO, -neg_KL, ll, mu, log_var, len(p)), fg='cyan')

    # Tests
    titles = ["$\mathcal{L}(\phi,\\theta;x)$",
              "$KL(q_{\phi}(z|x)||p_{\\theta}(z))$",
              "$\log(p_{\\theta}(x|z))$"]
    params = {"$\eta_{%s}$" % OPT["type"]: OPT["{}_rate".format(OPT["type"])],
              "$Activation$": ARCH["activation"],
              "$Batch$ $Size$": TRAIN["batch_size"],
              "$MCE$ $Samples$": PARAMS["L"],
              "$Latent$ $Dim$": DIMS["data"]}
    vis.basic_multiplot([iters]*3, [[ELBOs], [KLs], [LLs]], titles,
        show_legend=False, params=params)
    

# TODO: When implementating Variational Dropout, can use tf.nn.dropout
# TODO: Record error vectors

cl.secho('Success!', fg='green', bold=True)
