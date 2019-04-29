import sys
import os

sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from utils.topic_dists import dists_over_years
from visualization.latexify import format_axes


def distribution_per_year(df, path : str=None, display : bool=False):
    """

    """
    years, dists = dists_over_years(df)

    previous = dists[0, :]
    axis = range(dists.shape[1])

    fig, ax = plt.subplots(1, 1)
    ax.grid(linestyle='-.', color='lightgray')

    ax.bar(axis, previous, alpha=0.7, label=years[0])
    for i in range(1, dists.shape[0]):
        current = dists[i, :]
        ax.bar(axis, current, bottom=previous, label=years[i], alpha=0.7)
        previous = current

    #lgd = plt.legend()
    ax.grid(linestyle='-.', color='lightgray')
    ax.set_facecolor('white')

    format_axes(ax)
    if path:
        plt.savefig(os.path.join(path, 'dist_per_year.pdf'))
    if display:
        plt.show()
    plt.clf()
    plt.close()


def distribution_per_topic(df, path : str=None, display : bool=False):
    """

    """
    years, dists = dists_over_years(df)

    fig, ax = plt.subplots(1, 1, figsize=(4, 5.3))
    ax.grid(linestyle='-.', color='lightgray')

    previous = np.zeros((dists.shape[0], ))
    axis = pd.to_datetime(years, format='%Y')

    for i in range(0, dists.shape[1]):
        current = dists[:, i] + previous
        ax.plot(axis, current, alpha=0.5, linewidth=0.8)
        ax.fill_between(axis, current, previous, alpha=0.3)

        diff = current.max()-previous.max()
        if diff >= 0.025:
            plt.annotate(f'Topic {i}', xy=(pd.to_datetime(years[-1]), previous.max()+diff/2))

        previous = current

    plt.ylim(bottom=0, top=1)
    plt.xlim(left=axis[0], right=axis[-1])
    ax.set_facecolor('white')

    format_axes(ax)

    if path:
        plt.savefig(os.path.join(path, 'dist_per_topic.pdf'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
