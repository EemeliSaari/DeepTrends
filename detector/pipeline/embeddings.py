import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import normalize

from base import BaseModel


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

        if dictionary:
            self.obj.build_vocab([[v for v in dictionary.values()]])

        self.normalize = normalize
        self.batch_size = batch_size

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
        res = np.empty((len(X), self.obj.vector_size))
        for i, word in enumerate(X):
            res[i, :] = self.obj.wv[word]
        if self.normalize:
            return normalize(res, norm=self.normalize, axis=1)
        return res
