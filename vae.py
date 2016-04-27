import tensorflow as tf
import numpy as np


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

# Encoder construction (Gaussian) -- cf. [1]: Appendix C

# Inputs - placeholder for mini-batches of (flattened) images
x_batch = tf.placeholder(tf.float32, shape=[M, MNIST_FLAT_DIM])

# Weights
W_hidden_e = tf.Variable(tf.zeros([MNIST_FLAT_DIM, HIDDEN_LAYERS_ENC]))
W_mu_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))
W_log_sigma_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))

# Biases
b_hidden_e = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC]))
b_mu_q = tf.Variable(tf.zeros([J]))
b_log_sigma_q = tf.Variable(tf.zeros([J]))

# Activations
h_e = tf.tanh(  # TODO: try ReLU instead of tanh?
        tf.matmul(x_batch, W_hidden_e) + b_hidden_e)
mu_q = tf.reduce_mean( # Average mu's produced by each datapoint in minibatch
                       #    into single mu to parameterize latent approx. q(z)
        tf.matmul(h_e, W_mu_q) + b_mu_q, 0)
sigma_q = tf.sqrt(tf.exp( # exp/sqrt because this is log variance estimate
                          # TODO: Why do we use log variance? Do we expect
                          # extremely high/low values? Does it help with
                          # vanishing weights phenomenon?
        tf.reduce_mean(
            tf.matmul(h_e, W_log_sigma_q) + b_log_sigma_q, 0)))

# Decoder construction (Gaussian) -- cf. [1]: Appendix C

# Inputs - latent space samples w/ reparameterization (z = g(ep,x); ep ~ p(ep))
#   This outsources randomness to auxiliary variable (see [1]: Sec 2.4)
#   For a Gaussian 'q' we have: g(ep,x) = mu + sigma*ep, and p(ep) = N(0,I)
# Note: Element-wise univariate Gaussian sampling <=> multivariate Gaussian sampling
# TODO: Introduce L later here
ep = tf.random_normal([J], mean=0, stddev=1) # TODO: Do we resample this everytime?
                                             # TODO: Is TF resampling everytime?
z = tf.expand_dims(mu_q + sigma_q*ep, 0) # element-wise ops, expand to be 1xJ

# Weights
W_hidden_d = tf.Variable(tf.zeros([J, HIDDEN_LAYERS_DEC]))
W_mu_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))
W_log_sigma_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))

# Biases
b_hidden_d = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC]))
b_mu_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))
b_log_sigma_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))

# Activations
h_d = tf.tanh(  # TODO: try ReLU instead of tanh?
        tf.matmul(z, W_hidden_d) + b_hidden_d)
mu_p = tf.reduce_mean( # Average mu's produced by each Monte Carlo sample of q(z)
                       #    into single mu to parameterize generative p(x|z)
        tf.matmul(h_d, W_mu_p) + b_mu_p, 0)
sigma_p = tf.sqrt(tf.exp( # exp/sqrt because this is log variance estimate
                          # TODO: Why do we use log variance? Do we expect
                          # extremely high/low values? Does it help with
                          # vanishing weights phenomenon?
        tf.reduce_mean(
            tf.matmul(h_d, W_log_sigma_p) + b_log_sigma_p, 0)))

# ELBO estimator construction

KL_prior_regularizer = None # The KL divergence between the variational approx.
                            # and the prior p_theta(z) acts as a regularizing
                            # term so that the latent variables don't overfit.
                            # The closed-form eq. is derived in [1]: Appedix B

pred_reconstr_err =   None  # Measures log p_theta(x_i|z)
ELBO_estimate = KL_prior_regularizer + pred_reconst_err

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
