import tensorflow as tf


MNIST_FLAT_DIM = 784    # Flattened dimension of MNIST images, 'x'
HIDDEN_LAYERS_ENC = 500
HIDDEN_LAYERS_DEC = 200

# Experimental Parameters
M = 100     # minibatch size
L = 1       # MC samples
J = 10      # Dimensionality of latent space, 'z'

# Encoder construction (Gaussian) -- cf. [1]: Appendix C
W_hidden_e = tf.Variable(tf.zeros([MNIST_FLAT_DIM, HIDDEN_LAYERS_ENC]))
W_mu_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))
W_log_sigma_q = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC, J]))

b_hidden_e = tf.Variable(tf.zeros([HIDDEN_LAYERS_ENC]))
b_mu_q = tf.Variable(tf.zeros([J]))
b_log_sigma_q = tf.Variable(tf.zeros([J]))

# Decoder construction (Gaussian) -- cf. [1]: Appendix C
W_hidden_d = tf.Variable(tf.zeros([J, HIDDEN_LAYERS_DEC]))
W_mu_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))
W_log_sigma_p = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC, MNIST_FLAT_DIM]))

b_hidden_d = tf.Variable(tf.zeros([HIDDEN_LAYERS_DEC]))
b_mu_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))
b_log_sigma_p = tf.Variable(tf.zeros([MNIST_FLAT_DIM]))

print("Success!")
