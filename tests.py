#!/usr/bin/python

""" tests.py: Quantitative and qualitative tests on the effectiveness of the
        VAE implementation

    TODO:
        - Use nosetests-like framework
        - Auto-save to graphs/
        - Click CLI
"""

import tensorflow as tf
import click as cl
import numpy as np
import yaml

import vae
import vis


def simple_test():
    '''
    Train VAE using default parameters specified in config/nn_config.yaml and
    plot progress over training iterations
    '''
    fconf = 'config/nn_config.yaml'

    try:
        config = yaml.load(file(fconf, 'r'))
    except yaml.YAMLError, exc:
        cl.secho("Error in configuration file: {}".format(exc), fg='red')
        return

    ARCH   = config["architecture"]
    DIMS   = config["dims"]
    OPT    = config["optimization"]
    PARAMS = config["AEVB"]
    TRAIN  = config["training"]

    # TF backend initializations
    tf.set_random_seed(0)

    # Instantiate and train vanilla autoencoder
    nn = vae.VAE(ARCH, DIMS, OPT, PARAMS, TRAIN)
    results = nn.train()
    iters = results["iters"]
    ELBOs = results["ELBO"]
    KLs = results["KL"]
    LLs = results["LL"]

    # Graph training results
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


if __name__ == "__main__":
    simple_test() 
