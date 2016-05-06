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

### Statistical Optimizations
1. Xavier Initialization
2. Variational Dropout
5. Reparameterization I: Reducing MCE Variance

When performing Variational Inference, there are two approaches to finding the gradient of the lower bound (which is an intractable expectation over q(z)): (1) find an analytic (and computationally attractive) closed form of the gradient w.r.t. variational parameters, or (2) Monte-Carlo sampling 'z ~ q(z)' and using these z's to estimate the expectation of a function (in this case, a gradient). Unfortunately, (1) is oftentimes infeasible when we want to remove the constraint of conjugacy and (2) is troublesome when we naively perform expectation over the gradients of the random samples since this estimator exhibits high variance [citation].

As a side note, high variance is especially troublesome in a neural network setting where we there are 2 sources of variance for the true gradient: one from using a minibatch of data points (rather than the entire batch) to compute a gradient estimate (i.e. SGD), and another from calculating the gradient of the minibatch using Monte-Carlo methods (note, the second source only occurs when the inputs to the loss function are also stochastic, e.g. an expectation as in the ELBO). This "doubly stochastic" nature where both the inputs to the function as well as the function itself is stochastic can lead to problems when either source exhibits high variance.

To ameliorate this from a purely statistical point of view, we reformulate (2) not as a computing an expectation over gradients of single points, but rather a gradient on an expectation (of a function, g) over single points. The latter is only possible if we are able to (analytically) reparameterize 'z' into a random component (ep) and deterministic component (g) which includes the variables that the gradient is taken with respect to (ex: for a Gaussian, this would be mu/sigma) so that we can analytically compute the gradient w.r.t. phi of g(ep, phi) after summing samples of g(ep, phi). In other words, we must find a way to outsource the uncertainty on 'z' to an auxiliary variable (not dependent on phi) so we can still accurately leverage Monte Carlo sampling on the auxiliary variable (i.e. g(ep, phi), ep~p(ep) should give the same distribution as q(z;phi)) and also take a gradient of an MCE rather than an MCE of a gradient (thereby greatly reducing the variance).

Having approached this from a purely statistical point of view, we can expect better performance with evidence *grounded in theory*, whereas SGD is still missing substantial theoretical justification.

6. Reparameterization II: Reducing Variance at Scale
7. Streaming VB

### Numerical Optimizations
1. ADAM Optimizer

3. Batch Normalization
8. ReLU Activation
9. TF-Related Optimizations
10. Natural Gradients
11. Posterior Predictive Checking
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
