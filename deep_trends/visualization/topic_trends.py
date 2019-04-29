import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d

sys.path.append('..')

from utils.topic_dists import dists_over_years
from visualization.latexify import format_axes


register_matplotlib_converters()


def topic_dist_per_year(df, path : str=None, display : bool=False, 
                        smooth_method : str='gauss', **kwargs):
    years, dists = dists_over_years(df)
    axis = pd.to_datetime(years, format='%Y')

    for i in range(dists.shape[1]):
        ax = plt.subplot(111)
        ax.grid(linestyle='-.', color='lightgray')

        original = dists[:, i]
        if smooth_method == 'gauss':
            smooth = gaussian_filter1d(original, sigma=2)
        elif smooth_method == 'spline':
            spl = UnivariateSpline(x=range(original.shape[0]), y=original)
            smooth = [spl(i) for i in range(original.shape[0])]
        else:
            raise ValueError(f'Got unknown smoothing method: {smooth_method}')

        ax.plot(axis, smooth, linewidth=0.8, c='darkorange')
        ax.fill_between(axis, smooth, smooth-(smooth-original), alpha=0.3, color='darkorange')
        ax.set_facecolor('white')

        format_axes(ax)
        if path:
            plt.savefig(os.path.join(path, f'topic{i}_dist_per_topic_{smooth_method}.pdf'))
        if display:
            plt.show()
        plt.clf()
        plt.close()


def topic_regression(df, path : str=None, display : bool=False, 
                     truncate : bool=True, order : int=3, **kwargs):
    """

    """
    import seaborn

    years, dists = dists_over_years(df)
    data = dists.tolist()
    for seq, y in zip(data, years):
        seq.append(y)

    n_topics = dists.shape[1]
    dist_year_df = pd.DataFrame(data)
    dist_year_df.columns = [f'Topic-{i}' for i in range(n_topics)] + ['year']

    for i in range(n_topics):
        ax = seaborn.regplot('year', f'Topic-{i}', data=dist_year_df, order=3, 
        truncate=truncate)

        ax.grid(linestyle='-.', color='lightgray')
        ax.set_facecolor('white')

        format_axes(ax)
        if path:
            plt.savefig(os.path.join(path, f'topic{i}_dist_per_topic_{order}regression.pdf'))
        if display:
            plt.show()
        plt.clf()
        plt.close()
