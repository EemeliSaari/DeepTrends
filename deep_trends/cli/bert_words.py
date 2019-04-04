import random
import os
import sys
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mxnet as mx
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from bert_embedding import BertEmbedding

sys.path.append('..')

style.use('ggplot')

from corpora import Corpora

data_path = 'M:/Projects/KeyTopicDetection/parsed'
dict_path = '../../data/cvpr_13-18_DICT.txt'

if os.path.exists(dict_path):
    corpora = Corpora(data_path=data_path, prefix='CVPR', iterator='bow', dictionary=dict_path)
else:
    corpora = Corpora(data_path=data_path, prefix='CVPR', iterator='bow', word_up_limit=0.75, word_low_limit=20).build()
    corpora.dictionary.save_as_text(dict_path)

ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx)

def visualize_clusters(tw, data):
    db = DBSCAN(eps=0.5, min_samples=50).fit(data)
    samples_mask = np.zeros_like(db.labels_, dtype=bool)
    samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    plt.figure(figsize=(12,8))
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8, alpha=0.3)

        xy = data[class_member_mask & ~samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5, alpha=0.2)
    plt.title(f'Word {tw} 2 principal components (N={data.shape[0]})')
    plt.savefig(f'{tw}_{n_clusters_}c_{data.shape[0]}N.png')
    #plt.show()

def word_clusters(tw):
    my_sentences = []
    for i, s in enumerate(corpora.sentences()):
        for ss in s:
            if tw in ss:
                my_sentences.append(ss)
    N = len(my_sentences)
    vectors = []
    batch = []
    for i, sentence in enumerate(my_sentences):
        batch.append(sentence)
        if i == N - 1:
            pass
        elif len(batch) < 100:
            continue

        result = bert.embedding(batch)
        for s in result:
            for i, word in enumerate(s[1]):
                if word == tw:
                    vectors.append(s[2][i])
        batch = []
    target_matrix = np.stack(vectors)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(target_matrix)

    visualize_clusters(tw, reduced)

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

files = os.listdir()

for token in sorted(corpora.dictionary.dfs, key=lambda k: corpora.dictionary.dfs[k], reverse=True):
    word = corpora.dictionary[token]
    if corpora.dictionary.dfs[token] < 1000:
        break
    if any(word in f for f in files):
        continue
    print(f'Processing word: {word}')
    word_clusters(word)
