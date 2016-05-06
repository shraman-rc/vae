# VINNy

Fun with Variational Autoencoder neural nets. (Project submission for Tamara Broderick's "Bayesian Inference" (6.882) course at MIT).

Authors: Shraman Ray Chaudhuri

## Introduction

### What is a "Variational Autoencoder"?

A variational autoencoder is an encoder-decoder pair of neural networks that are designed to perform variational inference (originally on the latent space of an HMM-esque "coding" model). This means that the objective function being optimized in the ANN is (an estimate of) the evidence lower bound ("ELBO"), which in turn decreases the "KL-divergence" (a measure of the difference in information between the distributions, or "relative entropy") between an approximative distribution (q(z|x)) and the true posterior (p(z|x)). A quick (but technical) introduction to Variational Inference can be found in \[2\] and the auto-encoding framework for VI in \[1\].

### Why are Variational Autoencoders important?

\[Stay tuned\]

## Cool Stuff

1. Application to Dirichlet Processes?

    - Can we apply the reparameterization techniques to this AEVB framework?
        [Derivation]

## Optimizations

The vanilla VAE has lots of potential but there are a few shortcomings that make their convergence rate suboptimal with respect to the state-of-the-art. Here are improvements that have been applied (so far) to this repository:

1. Initialization
2. ADAM Optimizer
3. Batch Normalization
4. Variational Dropout
5. Reparameterization Trick II
6. Streaming VB
7. ReLU Activation
8. TF-Related Optimizations
9. Natural Gradients
10. Posterior Predictive Checking
    - Different error function than just log p(x|z) for specific models (inspired by Bayesian Checking for Topic Models)

## Future Work

\[Stay tuned\]

## References
\[1\] [_Auto-encoding Variational Bayes_](http://arxiv.org/abs/1312.6114); Kingma, Welling; NIPS 2014

\[2\] [_Variational Inference Lecture Notes_](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf); Blei; Princeton 2011

\[3\] [_Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_](http://arxiv.org/abs/1502.03167); Ioffe, Szegedy; JMLR 2015

\[4\] [_Streaming Variational Bayes_](http://papers.nips.cc/paper/4980-streaming-variational-bayes.pdf); Broderick, Boyd, Wibisono, Wilson, Jordan; NIPS 2013

\[5\] [_Variational Dropout and Local Reparameterization Trick_](http://arxiv.org/pdf/1506.02557v2.pdf); Kingma, Salimans, Welling; NIPS 2015

\[6\] [_Stochastic Variational Inference_](http://arxiv.org/pdf/1206.7051.pdf); Hoffman, Blei, Wang, Paisley; JMLR 2013
