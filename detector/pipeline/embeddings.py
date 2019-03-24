import os

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import normalize

from base import BaseModel
from corpora import Corpora


class Word2VecWrapper(BaseModel):
    """

    """
    obj = Word2Vec

    def __init__(self, weights : str = None, size : int=100, window : int=5,
                 min_count : int=1, normalize : str=None, dictionary=None,
                 batch_size : int=10, **kwargs):
        if weights:
            self.obj = Word2Vec.load_word2vec_format(weights, binary=True)
        else:
            super(Word2VecWrapper, self).__init__(
                size=size,
                window=5,
                min_count=1,
                **kwargs)

        self.__ws = window

        if dictionary:
            self.obj.build_vocab([[v for v in dictionary.values()]])

        self.normalize = normalize
        self.batch_size = batch_size

    def __str__(self):
        name = f'word2vec_{self.obj.vector_size}size_{self.__ws}win_{len(self.obj.wv.vocab)}N'
        if self.normalize:
            name += '_normalized'
        return name

    def fit(self, X, y=None, epochs=50):
        """

        """
        batch = []
        for i, (doc, N) in enumerate(X):
            batch.append(doc)
            if i == N - 1:
                pass
            elif len(batch) < self.batch_size:
                continue
            self.obj.train(
                batch, 
                total_examples=len(batch),
                epochs=epochs
            )
            batch = []

        self.fitted_ = True

        return self

    def transform(self, X):
        """

        """
        if isinstance(X, Corpora):
            words = X.dictionary.values()
        elif isinstance(X, (tuple, list)):
            words = X

        vectors = np.empty((len(words), self.obj.vector_size))
        for i, word in enumerate(words):
            vectors[i, :] = self.obj.wv[word]
        if self.normalize:
            vectors = normalize(vectors, norm=self.normalize, axis=1)

        res = {w:vectors[i, :] for i, w in enumerate(words)}

        return res

    @property
    def vectors(self):
        """

        """
        if self.normalize:
            vectors = np.empty((len(self.obj.wv), self.obj.vector_size))
            for i, word in enumerate(self.obj.wv):
                vectors[i, :] = self.obj.wv[word]
            vectors = normalize(vectors, norm=self.normalize, axis=1)
            return {w:vectors[i, :] for i, w in enumerate(self.obj.wv)}
        return self.obj.wv


def load_vectors(path, int_columns=True, dictionary=None):
    """

    """
    if not path:
        return
    if not os.path.exists(path):
        raise OSError()
    df = pd.read_csv(path)
    if dictionary:
        missing = list(filter(lambda w: w not in dictionary.values(), df.columns))
        df = df.drop(missing, axis=1)
        df.columns = [dictionary.token2id[word] for word in df.columns]

    if int_columns and not dictionary:
        df.columns = [int(c) for c in df.columns]
    return df
