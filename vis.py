""" vis.py: Visualization tools for demos and interactive UI
"""

__author__ = 'shraman-rc'

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as widget
import numpy as np

def basic_multiplot(data_xs, data_ys, titles, labels=None, unit_x="Minibatches", show_legend=True, params={}):
    """ Compare multiple plots (e.g. different error signals)

        - data_xs: List of nd.arrays for x-axis for each plot
        - data_ys: List of lists of nd.arrays for y-axis for each plot
        - titles: List of strings, one for each plot
        - labels: List of lists of strings, one for each (x,y) pair
        - unit_x: Shared units for x-axis (e.g. epochs, minibatches)
    """
    num_plots = len(data_xs)
    fig, axes = plt.subplots(num_plots)
    if num_plots == 1:
        axes = [axes]

    # Looks
    plt.rc('axes', color_cycle=['g', 'm', 'k', 'c'])
    plt.tight_layout() # So axis no overlap with title

    if not show_legend:
        labels = [['']*len(arrs) for arrs in data_ys]
    else:
        assert(labels)

    # Plot each line in each subplot
    for i, x in enumerate(data_xs):
        for y, l in zip(data_ys[i], labels[i]):
            axes[i].plot(x, y, label=l)
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_ylabel("Values", fontsize=16)
        axes[i].grid()
        if show_legend:
            axes[i].legend(loc="upper right",
                ncol=1,
                shadow=True,
                title="Heuristics",
                fancybox=True,
                prop={'size':15})

    # Label common x-axis
    axes[-1].set_xlabel("{}".format(unit_x), fontsize=16)

    # Parameter legend
    paramtex = '\n'.join(['${}={}$'.format(k,v) for k,v in params.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    axes[0].text(0.95, 0.10, paramtex, transform=axes[0].transAxes, fontsize=16,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.show()


def basic_multiline(data_x, data_ys, x_axis="Minibatch", y_axis="Error", title="Convergence Rate of ELBO"):

    for data_y in data_ys:
        line = plt.plot(data_x, data_y)[0]
        line.set_linewidth(2.0)

    plt.legend(loc="upper left", ncol=1, shadow=True, title="Errors", fancybox=True, prop={'size':25})
    plt.title(title, fontsize=30)
    #plt.xscale("log")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    
    plt.show()
