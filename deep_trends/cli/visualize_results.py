import sys
import os

sys.path.append('..')

import click
import seaborn
import pandas as pd
from matplotlib import style

from visualization.overall_distributions import distribution_per_topic, distribution_per_year
from visualization.topic_trends import topic_dist_per_year, topic_regression


@click.command()
@click.option('--path', '-p', type=str)
@click.option('--words', '-w', type=str)
@click.option('--results', type=str)
@click.option('--display', is_flag=True)
def main(path, words, results, display):

    if path.endswith('.csv'):
        df = pd.read_csv(path)
        words_df = pd.read_csv(words)
    else:
        df = pd.read_csv(os.path.join(path, 'topics.csv'))
        words_df = pd.read_csv(os.path.join(path, 'words.csv'))

    style.use('ggplot')
    seaborn.set_style('whitegrid')

    if not results:
        results = path

    distribution_per_topic(df, path=results, display=display)
    distribution_per_year(df, path=results, display=display)
    paths_map = {
        'spline': topic_dist_per_year, 
        'gauss':topic_dist_per_year, 
        'regression': topic_regression
    }
    for path, func in paths_map.items():
        dir_path = os.path.join(results, path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        func(df, path=dir_path, display=display, smooth_method=path)

    for col in words_df.columns:
        print(col, words_df[col].values)


def go():
    main()


if __name__ == '__main__':
    go()
