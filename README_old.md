# VINNy

Fun with Variational Autoencoder neural nets.

Ping me with any questions at shraman (at) mit (dot) edu! This repo will be continually updated with some cool (experimental) uses for variational inference performed by neural networks.

This repo is also a project submission for Tamara Broderick's "Bayesian Inference" (6.882) course at MIT.

## Introduction

### What is a "Variational Autoencoder"?

A variational autoencoder is an encoder-decoder pair of neural networks that are designed to perform approximate posterior inference on latent variables. The loss function is (an estimate of) the evidence lower bound, optimizing which decreases the KL-divergence (or "relative entropy") between the approximating and true posteriors. A quick introduction to VI can be found in \[2\] and the auto-encoding framework for VI in \[1\].

## Optimizations

### Statistical Optimizations
#### Reparameterization I: Reducing MCE Variance

In my opinion, this is the coolest aspect of [1] despite being so simple, since it brings out the ability to achieve good results with AEVB, but at a larger scale, broaches a new methodology altogether in approximation posterior inference with Variational Inference that depends on Monte Carlo estimation to project VI onto a broader class of models. It is definitely worth understanding this reparameterization trick in depth -- motivations, derivations, and all.

When performing Variational Inference, there are two approaches to finding the gradient of the lower bound (which is an intractable expectation over q(z)): (1) find an analytic (and computationally attractive) closed form of the gradient w.r.t. variational parameters, or (2) Monte-Carlo Estimation (MCE): sampling 'z ~ q(z)' and using these samples z^(l) to somehow estimate the expectation of a function (in this case, a gradient of an expectation). Unfortunately, (1) is oftentimes infeasible when we want to remove the constraint of conjugacy between the prior and likelihood and (2) is troublesome when we naively perform expectation over the gradients of the random samples since this estimator exhibits high variance [citation].

As a side note, high variance is especially troublesome in a neural network setting where there are 2 sources of variance for the true gradient: one from using a random minibatch of data points (rather than the entire batch) to compute a gradient estimate (i.e. SGD), and another from calculating the gradient of that minibatch using Monte-Carlo methods (which we only use when the *inputs* to the loss function are also stochastic, e.g. when the loss function is an expectation like the variational lower bound). This "doubly stochastic" nature where both the inputs to the function as well as the function itself is stochastic can lead to problems when either source exhibits high variance.

To ameliorate this from a purely statistical point of view, we reformulate (2) not as a computing an expectation over gradients of single points, but rather a gradient on an expectation (of a function, g) over single points. The latter is only possible if we are able to (analytically) reparameterize 'z' into a random component (ep) independent of phi (i.e. the variable with respect to which we are taking the gradient) and deterministic component (g) which includes phi (ex: for a Gaussian, this would be mu/sigma) so that we can analytically compute the gradient (w.r.t. phi) of g(ep, phi) after summing MCE samples of g(ep, phi). In other words, we must find a way to outsource the uncertainty on 'z' to an auxiliary variable (not dependent on phi) so we can still accurately leverage Monte Carlo sampling on the auxiliary variable (i.e. g(ep, phi), ep~p(ep) should give the same distribution as q(z; phi)) and while also being able to take a gradient of an MCE rather than an rely on an MCE of a gradient (thereby greatly reducing the variance).

Having approached this from a purely statistical point of view, we can expect better performance with evidence *grounded in theory*, whereas SGD is still missing substantial theoretical justification. This reparameterization also motivates a cool way to incorporate MCMC into Variational Inference to produce even better approximations, results of which can be tuned to tradeoff time (longer MCMC) with better results [7].

#### Reparameterization II: Reducing Variance at Scale

#### Variational Dropout

#### Natural Gradients

#### Xavier Initialization

#### Streaming VB

#### Posterior Predictive Checking

### Numerical Optimizations
#### ADAM Optimizer

#### Batch Normalization

#### Activation Functions


## References
\[1\] [_Auto-encoding Variational Bayes_](http://arxiv.org/abs/1312.6114); Kingma, Welling; NIPS 2014

\[2\] [_Variational Inference Lecture Notes_](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf); Blei; Princeton 2011

\[3\] [_Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_](http://arxiv.org/abs/1502.03167); Ioffe, Szegedy; JMLR 2015

\[4\] [_Streaming Variational Bayes_](http://papers.nips.cc/paper/4980-streaming-variational-bayes.pdf); Broderick, Boyd, Wibisono, Wilson, Jordan; NIPS 2013

\[5\] [_Variational Dropout and Local Reparameterization Trick_](http://arxiv.org/pdf/1506.02557v2.pdf); Kingma, Salimans, Welling; NIPS 2015

\[6\] [_Stochastic Variational Inference_](http://arxiv.org/pdf/1206.7051.pdf); Hoffman, Blei, Wang, Paisley; JMLR 2013

\[7\] [_MCMC and Variational Inference: Bridging the Gap_](http://arxiv.org/pdf/1410.6460v4.pdf); Salimans, Kingma, Welling; JMLR 2015
