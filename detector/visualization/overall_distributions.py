import sys
import os

sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from utils.topic_dists import dists_over_years


def distribution_per_year(df, path : str=None, display : bool=False):
    """

    """
    years, dists = dists_over_years(df)

    plt.figure(figsize=(20,12))

    previous = dists[0, :]
    axis = range(dists.shape[1])

    plt.bar(axis, previous, alpha=0.8, label=years[0])
    for i in range(1, dists.shape[0]):
        current = dists[i, :]
        alpha = 0.3
        if current.max() > .3:
            alpha = 1
        plt.bar(axis, current, bottom=previous, label=years[i], alpha=alpha)
        previous = current
    plt.legend()
    if path:
        plt.savefig(os.path.join(path, 'dist_per_year.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()


def distribution_per_topic(df, path : str=None, display : bool=False):
    """

    """
    years, dists = dists_over_years(df)

    plt.figure(figsize=(8,12))

    previous = np.zeros((dists.shape[0], ))
    axis = pd.to_datetime(years, format='%Y')

    for i in range(0, dists.shape[1]):
        current = dists[:, i] + previous
        plt.plot(axis, current, alpha=0.5)
        plt.fill_between(axis, current, previous, alpha=0.3)

        diff = current.max()-previous.max()
        if diff >= 0.025:
            plt.annotate(f'Topic {i}', xy=(pd.to_datetime(years[-1]), previous.max()+diff/2))

        previous = current

    plt.ylim(bottom=0, top=1)
    plt.xlim(left=axis[0], right=axis[-1])
    if path:
        plt.savefig(os.path.join(path, 'dist_per_topic.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
