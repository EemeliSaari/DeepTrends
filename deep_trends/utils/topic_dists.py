import numpy as np


def dists_over_years(df):
    years = df['year'].unique()

    dists = []
    for y in years:
        data = df.where(df.year == y).dropna().drop(['year', 'doc'], axis=1).values
        dists.append(data.sum(axis=0)/data.shape[0])

    return years, np.array(dists)
