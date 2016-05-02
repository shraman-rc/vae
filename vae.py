import tensorflow as tf
import numpy as np

from nets import BernoulliMLP, GaussianMLP
import likelihoods as lh

MNIST_FLAT_DIM = 784    # Flattened dimension of MNIST images, 'x'
HIDDEN_LAYERS_ENC = 500
HIDDEN_LAYERS_DEC = 200


# Experimental Parameters
M = 100         # minibatch size
L = 1           # MCE samples
J = 10          # Dimensionality of latent space, 'z'
T = 10000       # Training iterations
ADG_RATE = 0.01 # Adagrad global learning rate (chosen from
                #   {0.01,0.02,0.1} -- cf. [1]: Section 5)

# Inputs - mini-batches of (flattened) images
x_batch = tf.placeholder(tf.float32, shape=[M, MNIST_FLAT_DIM])

encoder = GaussianMLP(x_batch) # Gaussian q(z|x)

# Bridge between encoder and decoder
# Latent space samples w/ reparameterization (z = g(ep,x); ep ~ p(ep))
#   This outsources randomness to auxiliary r.v. (see [1]: Sec 2.4)
#   For a Gaussian 'q' we have: g(ep,x) = mu + sigma*ep, and p(ep) = N(0,I)
# Note: Element-wise univariate Gaussian sampling <=> multivariate Gaussian sampling
# TODO: Introduce L later here
ep = tf.random_normal([M, J], mean=0, stddev=1) # TODO: Is TF resampling everytime?
z_batch = mu_q + sigma_q*ep # element-wise ops, z_batch has latent per datapoint (MxJ)

decoder = BernoulliMLP(z_batch)

#
## Weights
#W_hidden_e = tf.Variable(tf.zeros([MNIST_FLAT_DIM, HIDDEN_LAYERS_ENC]))
#W_mu_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))
#W_log_sigma_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))
#
## Biases
#b_hidden_e = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC]))
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
#sigma_sq_q = tf.exp(log_sigma_sq_q)
#sigma_q = tf.sqrt(sigma_sq_q)

# Decoder construction (Bernoulli, for MNIST)

## Weights
#W_hidden_d = tf.Variable(tf.zeros([J, HIDDEN_LAYERS_DEC]))
#W_mu_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))
#W_log_sigma_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))
#
## Biases
#b_hidden_d = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC]))
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

# ELBO estimator construction
sigma_p = tf.sqrt(tf.exp(log_sigma_sq_p))

# The KL divergence between the variational approx. and the prior p_theta(z) acts as
#   a regularizing term so that the latent variables don't overfit. The closed-form
#   eq. is derived in [1]: Appedix B
# TODO: Why would overfitting be a problem in the auto-encoding scenario? Wouldn't
#   overfitting lead to a better likelihood lower bound measure used in [1]'s experiments?
KL_prior_regularizer = 0.5 * tf.reduce_sum(1 + log_sigma_sq_q - mu_q - sigma_sq_q, 1)
pred_reconstr_err = lh.ll_bernoulli()  # Measures log p_theta(x_i|z) # TODO: Try a different loss function? Cross entropy?

# ^ TODO: CHANGE TO BERNOULLI AND COMPARTMENTALIZE CODE (backends.py) WE TAKE ELEMENT WISE SIGMOID TO PARAMETERIZE
# MULTIVARIATE BERNOULLI (i.e. pixel values in image - note will need to rescale if doing on regular images) BUT
# HOW DOES STEEPNESS AFFECT LEARNING?

# TODO: Apply batch normalization here as in [3]? See effects on Adagrad, SGD, and ADAM separately?
ELBO_estimate = tf.reduce_mean(KL_prior_regularizer + pred_reconst_err) # Mean val over minibatch

# TODO: Using Adagrad as per [1] but was written before ADAM (by same author!)
#       Later transition to ADAM
# Notice that we are maximizing (not minimizing) the variational lower bound
train_op = tf.train.Adagrad(ADG_RATE).maximize(ELBO_estimate)

# Train on MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for _ in range(T):
    pass 

print("Success!")
