import collections
import logging
import os
import time
import re

import numpy as np
import pandas as pd
from filelock import FileLock
from gensim.parsing import preprocess_string
from pipeline.preprocess import custom_preprocess
from gensim.corpora.dictionary import Dictionary


class Corpora:
    """

    """
    dictionary = Dictionary()
    is_built = False

    def __init__(self, data_path : str, prefix : str=None, 
                 iterator : str='token', vector_map_path : str=None, 
                 parsing : str='simple', shuffle : bool=False,
                 seed : int=42):

        iter_map = dict(
            token=self.tokenize,
            glovize=self.glovize,
            bow=self.bowize
        )
        self.iterator = iter_map[iterator]
        self.corpus = self.init_corpus(data_path, prefix, parsing)

        if vector_map_path:
            self.vector_map = self.load_vectors(vector_map_path)

        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)

    def __enter__(self):
        self.build()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def __iter__(self):
        for v in self.iterator():
            yield v

    def __len__(self):
        return sum([c.n_docs for c in self.corpus])

    def init_corpus(self, path : str, prefix : str, parsing : str):
        """

        """
        directory = [os.path.join(path, f) for f in os.listdir(path)]
        folders = list(filter(lambda p: os.path.isdir(p), directory))
        if prefix:
            folders = list(filter(lambda p: prefix in p, folders))

        return [Corpus(path=p, parsing=parsing).load() for p in folders]

    def load_vectors(self, path : str):
        """

        """
        if not path.endswith('.csv'):
            raise AssertionError('Asserted the vectors to be provided with csv.')
        #TODO Use dask in case of too large word vector maps.
        return pd.read_csv(path)

    def build(self):
        """

        """
        if self.is_built:
            logging.warn('Attempted to build already built Corpora.')
            return

        for c in self.corpus:
            self.dictionary.add_documents(c.tokens)
            c.clear()

    def clear(self):
        """

        """
        self.dictionary = Dictionary()

    def bowize(self):
        """

        """
        for doc_tokens, N in self.tokenize():
            yield self.dictionary.doc2bow(doc_tokens), N

    def glovize(self):
        if not hasattr(self, 'vector_map'):
            raise AssertionError('Cant glovize without vector map.')

        for i, (doc_bow, N) in enumerate(self.bowize()):
            bow_vectors = [(self.vector_map[self.dictionary[v[0]]].values, v[1]) for v in doc_bow]
            yield (np.array(bow_vectors), i), N

    def tokenize(self):
        """

        """
        if self.shuffle:
            np.random.shuffle(self.corpus)

        for c in self.corpus:
            for doc_tokens in c.tokens:
                yield doc_tokens, len(self)
            c.clear()

    @property
    def years(self):
        """

        """
        return sorted([int(c.year) for c in self.corpus])


class Corpus:

    PARSERS = dict(gensim=preprocess_string, simple=custom_preprocess)
    n_docs = 0
    is_loaded = False

    def __init__(self, path : str, parsing : str='gensim', 
                 encoding : str='utf-8', keep_tokens : bool=False,
                 keep_documents : bool=True):
        self.path = path
        self.encoding = encoding
        self.keep_tokens = keep_tokens
        self.keep_documents = keep_documents

        self.__parsing_method = parsing
        self.__tokens = []
        self.__docs = []

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(loaded={self.is_loaded}, n_documents={self.n_docs})'

    def load(self):
        if os.path.isdir(self.path):
            self.__documents = os.listdir(self.path)
            self.__docs = list(self._read_folder())
        else:
            self.__documents = [self.path]
            with FileLock(self.path):
                self.__docs = [self._read_file(self.path)]
        self.n_docs = len(self.__docs)
        self.is_loaded = True

        return self

    def clear(self):
        
        if not self.keep_documents:
            self.n_docs = 0
            self.__docs = []

        if not self.keep_tokens:
            self.__tokens = []

    @property
    def documents(self):
        return self.__docs

    @property
    def tokens(self):
        start = time.time()
        self.__tokens = list(map(self.PARSERS[self.__parsing_method], self.documents))
        logging.info('Corpora tokens took: {:.5f}s'.format(time.time()-start))
        return self.__tokens

    @property
    def id2doc(self):
        return self.__documents

    @property
    def year(self):
        return re.findall('\d+', self.path)[-1]

    def _read_folder(self):
        return map(lambda f: self._read_file(os.path.join(self.path, f)), self.__documents)

    def _read_file(self, path):
        with open(path, 'r', encoding=self.encoding) as f:
            return ''.join(f.readlines())
