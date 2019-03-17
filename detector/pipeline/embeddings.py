import numpy as np
from gensim.models.word2vec import Word2Vec

from base import BaseModel


class Word2VecWrapper(BaseModel):
    """

    """
    obj = Word2Vec

    def __init__(self, weights : str = None, size : int=100, window : int=5,
                 min_count : int=1, normalize : str=None, dictionary=None, 
                 **kwargs):
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

    def fit(self, X, y=None, epochs=1):
        """

        """
        for doc, N in X:
            self.obj.train(
                doc, 
                total_examples=len(doc),
                epochs=epochs
            )

        self.fitted_ = True

        return self

    def transform(self, X):
        """

        """
        res = np.empty((len(X), self.obj.vector_size))
        for i, word in enumerate(X):
            res[i, :] = self.obj.wv[word]
        return res
