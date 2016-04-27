# VINNy
Fun with Variational Autoencoder neural nets.

## Introduction

### What is a "Variational Autoencoder"?

A variational autoencoder is an encoder-decoder pair of neural networks that are designed to perform variational inference. This means that the objective function that the neural network is trying to optimize is (an estimate of) the evidence lower bound ("ELBO"), which in turn decreases the "KL-divergence" (a measure of the difference in information between the distributions, or "relative entropy") between an approximative distribution (q(z|x)) and the true posterior (p(z|x)). A quick (but technical) introduction to Variational Inference can be found in \[2\] and the auto-encoding framework for VI in \[1\].

### What can we do with Variational Autoencoders?

Lots of cool things. Stay tuned.

## References
\[1\] [_Auto-encoding Variational Bayes_](http://arxiv.org/abs/1312.6114); Kingma, Welling; 2014

\[2\] [_Variational Inference Lecture Notes_](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf); Blei; 2011
