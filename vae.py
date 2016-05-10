""" vae.py: Run the Variational Auto-encoder.

    TODO:
        - Incorporate L (MC samples) without blowing up the decoder variable count
        - There's a blow-up of the log_var output of the encoder which makes
            the KL-divergence term of the error function go to infinity since there
            is a var term (where var = e^{log_var}). This seems to be irreversible
            when the learning rate is high.

            UPDATE: The network weights seem to be such that at the beginning, there
            is high variance output (exploding gradients when NN still malleable)

            UPDATE: Also exacerbated by batch size. Gradient clipping ineffective
            when gradient becomes nan!

            UPDATE: This problem is *super* sensitive to learning rate and highly
            nondeterministic. At a learning of 0.016 (using Adam) gradients will
            sometimes blow up to 3e+28 and when it doesn't, they will go no higher
            than 100! However, there is much more stability across runs even at 0.015!

            TODO: Try value clipping of the KL divergence. Norm clipping too?
        - We use output of sigmoid to parameterize the multivariate Bernoulli.
            Does its steepeness affect learning?
        - Apply batch normalization as in [3]
        - Try different reconstruction loss instead of log-likelihood:
            - Cross entropy
        - Quantitatively assess Adagrad, SGD, Adam performance
        - Use tf.nn.dropout to perform Variational Dropout
        - Examples:
            - DRAW network
            - Generative adversarial network
            - Music composition network (similar to DRAW)
            - Artistic network (similar to DRAW)
"""

__author__ = "shraman-rc"

import tensorflow as tf
import numpy as np
import click as cl

from nets import BernoulliMLP, GaussianMLP
import likelihoods as lh


tf.set_random_seed(0)

class VAE(object):

    def __init__(self, ARCH, DIMS, OPT, PARAMS, TRAIN):
        '''
        Initializes VAE with parameter dictionaries that should follow the
            same format as config/nn_config.yaml
        '''
        self.ARCH, self.DIMS, self.OPT, self.PARAMS, self.TRAIN = \
            ARCH, DIMS, OPT, PARAMS, TRAIN

        # Inputs - mini-batches of (flattened) images
        self.x_batch = tf.placeholder(tf.float32,
            shape=[None, self.DIMS["data"]])

        # Encoder parameterizes posterior Gaussian approximation q(z|x)
        self.encoder = GaussianMLP(self.x_batch,
            self.ARCH["encoder"]["n_units"], self.DIMS["latent"])
        self.latent = {}
        self.latent["mu"] = self.encoder.out_params.mu
        self.latent["log_var"] = self.encoder.out_params.log_var
        self.latent["var"] = tf.exp(self.latent["log_var"])
        self.latent["stddev"] = tf.sqrt(self.latent["var"])

        # Reparameterize latent space (z = g(ep,x); ep ~ p(ep))
        #   Note: Element-wise univariate Gaussian sampling <=>
        #         multivariate Gaussian sampling
        self.ep = tf.random_normal([self.TRAIN["batch_size"],
            self.DIMS["latent"]], mean=0, stddev=1)
        self.z_batch = self.latent["mu"] + self.latent["stddev"]*self.ep

        # Decoder samples from latent distribution, parameterizes likelihood
        #   (in this case, a multivariate Bernoulli - working with images)
        self.decoder = BernoulliMLP(self.z_batch,
            self.ARCH["decoder"]["n_units"], self.DIMS["data"])

        # The (negative) KL divergence between the variational approx. and the *prior*
        #   p_theta(z) acts as a regularizing term so that the latent distribution
        #   doesn't overfit. The closed-form eq. is derived in [1]: Appedix B
        self.neg_KL_pr = 0.5 * tf.reduce_sum(1 + self.latent["log_var"]
            - self.latent["mu"]**2 - self.latent["var"], 1)

        # The 'reconstruction error' (predictive likelihood): log p_theta(x_batch|z)
        self.ll = lh.ll_bernoulli(self.x_batch, self.decoder.out_params.p)

        # ELBO estimate (total reward function)
        self.ELBO_est = tf.reduce_mean(self.neg_KL_pr + self.ll) # Mean over minibatch

        # Pick a flavor of gradient descent
        if self.OPT["type"].lower() == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(self.OPT["Adagrad_rate"])
        elif self.OPT["type"].lower() == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.OPT["Adam_rate"])

        # Notice that we are minimizing the negative (i.e. maximizing) the ELBO
        #   We also clip the gradients to prevent blowup during first few
        #   training phases
        #self.train_op = self.optimizer.minimize(-self.ELBO_est)
        #self.vi_train_op = self.optimizer.minimize(-self.neg_KL_pr)
        #self.ll_train_op = self.optimizer.minimize(-self.ll)
        # ...with clipped gradients:
        gvs = self.optimizer.compute_gradients(-self.ELBO_est)
        capped_gvs = [(tf.clip_by_value(
            grad, -self.OPT["max_grad"], self.OPT["max_grad"]), var)
            for grad, var in gvs if grad != None]
        flat_grads = [tf.reshape(grad,[-1]) for grad, var in capped_gvs]
        self.max_grad = tf.reduce_max(tf.concat(0, flat_grads))
        self.train_op = self.optimizer.apply_gradients(capped_gvs)


    def _train_step(self, sess, data, verbose=True):
        ''' Common helper to run individual training steps, see _train()
            Returns output of salient variables above (e.g. ELBO) after
            each optimization iteration

            Params:
                - sess,verbose: See _train()
                - data: batch of raw data with correct dimensions
        '''
        _, ELBO, ll, neg_KL, mu, log_var, ep, max_grad = sess.run([
            self.train_op,
            self.ELBO_est,
            self.ll,
            self.neg_KL_pr,
            self.latent["mu"],
            self.latent["log_var"],
            self.ep,
            self.max_grad],
        feed_dict={self.x_batch: data})
        # _, ELBO, ll, neg_KL, mu, log_var, ep, max_grad = sess.run([
        #    self.vi_train_op,
        #    self.ELBO_est,
        #    self.ll,
        #    self.neg_KL_pr,
        #    self.latent["mu"],
        #    self.latent["log_var"],
        #    self.ep,
        #    self.max_grad],
        # feed_dict={x_batch: data})

        # Perform some sort of reductions on minibatches if need be
        ELBO, neg_KL, ll, mu, log_var, ep = (
            np.mean(ELBO),
            np.mean(neg_KL),
            np.mean(ll),
            mu[0],
            log_var[0],
            ep[0])

        # Print stats
        if verbose:
            cl.secho((
                "ELBO (estimate): {}\n"
                "KL Div (prior): {}\n"
                "Likelihood: {}\n"
                "Mu: {}\n"
                "Log var: {}\n"
                "Epsilon: {}\n"
                "Max grad: {}")
            .format(ELBO, -neg_KL, ll, mu, log_var, ep, max_grad), fg='cyan')

        return ELBO, ll, neg_KL, mu, log_var, ep, max_grad


    def _train(self, iters, mbsize, sess, verbose=True):
        ''' Trains the VAE end-to-end on MNIST (handwriting) dataset
            Returns progress through training phases on above variables
            via numpy arrays.
            
            Params:
                - iters: Number of training iteration per epoch
                - mbsize: Number of datapoints per minibatch
                - sess: TF session to use if already instantiated one
                    if None, will use temporary session
                - verbose: Print training progress after each timestep
        '''
        # Train on MNIST
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        # To keep track of progress
        progress = {}
        progress["ELBO"] = np.zeros(iters)
        progress["KL"] = np.zeros(iters)
        progress["LL"] = np.zeros(iters)

        # Optimize VAE
        sess.run(tf.initialize_all_variables())
        timesteps = xrange(iters)
        for t in timesteps:
            cl.secho('Minibatch {}'.format(t), fg='green', bold=False)
            batch = mnist.train.next_batch(mbsize)[0]
            prog = self._train_step(sess, batch, verbose)
            progress["ELBO"][t] = prog[0]
            progress["LL"][t] = prog[1]
            progress["KL"][t] = -prog[2]

        progress["iters"] = timesteps

        cl.secho('Success!', fg='green', bold=True)
        return progress


    def train(self, iters=None, mbsize=None, sess=None, verbose=True):
        ''' Wrapper for above _train() function
        '''
        iters = iters or self.TRAIN["n_iters"]
        mbsize = mbsize or self.TRAIN["batch_size"]

        if sess:
            progress = self._train(iters, mbsize, sess, verbose)    
        else:
            with tf.Session() as sess:
                progress = self._train(iters, mbsize, sess, verbose)

        return progress
