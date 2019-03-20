import collections
import itertools
import logging
import os
import re
import time

import numpy as np
import pandas as pd
from filelock import FileLock
from gensim.corpora.dictionary import Dictionary
from gensim.parsing import preprocess_string
from nltk import tokenize

from pipeline.preprocess import custom_preprocess


class Corpora:
    """

    """
    is_built = False

    def __init__(self, data_path : str, prefix : str=None, 
                 iterator : str='token', parsing : str='simple',
                 word_up_limit : float=0.5, word_low_limit : int=5,
                 dictionary : str=None, shuffle : bool=False, seed : int=42,
                 document_minimum_length : int=5):

        iter_map = dict(
            token=self.tokenize,
            bow=self.bowize
        )
        self.iterator = iter_map[iterator]
        self.corpus = self.init_corpus(data_path, prefix, parsing)

        self.word_low_limit = word_low_limit
        self.word_up_limit = word_up_limit
        if not dictionary:
            self.dictionary = Dictionary()
        else:
            self.dictionary = Dictionary.load_from_text(dictionary)
            self.is_built = True

        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)

        self.document_minimum_length = document_minimum_length

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

    def __getitem__(self, key):
        return self.iterator(index=key)

    def init_corpus(self, path : str, prefix : str, parsing : str):
        """

        """
        directory = [os.path.join(path, f) for f in os.listdir(path)]
        folders = list(filter(lambda p: os.path.isdir(p), directory))
        if prefix:
            folders = list(filter(lambda p: prefix in p, folders))

        corpus = [Corpus(path=p, parsing=parsing).load() for p in folders]
        self.__paths = {c.path: c for c in corpus}

        return corpus

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

        self.dictionary.filter_extremes(no_below=self.word_low_limit, no_above=self.word_up_limit)

        return self

    def clear(self):
        """

        """
        self.dictionary = Dictionary()

    def bowize(self, index=None):
        """

        """
        for doc_tokens, N in self.tokenize(index=index):
            yield self.dictionary.doc2bow(doc_tokens), N

    def tokenize(self, index=None):
        """

        """
        N = len(self)

        iterable = self._iterator(index)

        for idx in self._indices(iterable):
            corpus = iterable[idx]
            tokens = corpus.tokens
            for ind in self._indices(tokens):
                doc_tokens = tokens[ind]
                if len(doc_tokens) > self.document_minimum_length:
                    yield doc_tokens, N
                else:
                    logging.warn(f'Received empty file at {corpus.documents[ind]}, skipping.')
            corpus.clear()

    def sentences(self, index=None):
        """

        """
        iterable = self._iterator(index=index)
        for ind in self._indices(iterable=iterable):
            corpus = iterable[ind]
            for sentence in corpus.sentences:
                if len(sentence) > self.document_minimum_length:
                    yield sentence
                else:
                    logging.warn(f'Received empty file at {corpus.documents[ind]}, skipping.')

    def documents(self, index=None):
        """

        """
        for c in self.corpus:
            if len(c) > 1:
                yield c.documents
            else:
                for doc in c.documents:
                    yield doc

    @property
    def years(self):
        """

        """
        return sorted([int(c.year) for c in self.corpus])

    def _iterator(self, index=None):
        iterator = self.corpus
        if index:
            if isinstance(index, int):
                iterator = [self.corpus[index]] #TODO: Handle indices as slice
            elif isinstance(index, str):
                iterator = [self.__paths[index]]
        return iterator

    def _indices(self, iterable):
        """

        """
        if self.shuffle:
            indices = np.random.permutation(len(iterable))
        else:
            indices = range(len(iterable))
        return indices


class Corpus:
    """

    """
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
        """

        """
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
        """

        """
        if not self.keep_documents:
            self.n_docs = 0
            self.__docs = []

        if not self.keep_tokens:
            self.__tokens = []

    @property
    def raw(self):
        """

        """
        return self.__docs

    @property
    def sentences(self):
        """

        """
        too_short = lambda s: len(s.split(' ')) > 3
        texts = [' '.join(filter(too_short, l.split('\n'))) for l in self.raw]
        return list(map(tokenize.sent_tokenize, texts))

    @property
    def documents(self):
        """

        """
        return self.__documents

    @property
    def tokens(self):
        """

        """
        start = time.time()
        self.__tokens = list(map(self.PARSERS[self.__parsing_method], self.raw))
        logging.info('Corpora tokens took: {:.5f}s'.format(time.time()-start))
        return self.__tokens

    @property
    def id2doc(self):
        """

        """
        return self.__documents

    @property
    def year(self):
        """

        """
        return re.findall('\d+', self.path)[-1]

    def _read_folder(self):
        """

        """
        return map(lambda f: self._read_file(os.path.join(self.path, f)), self.__documents)

    def _read_file(self, path):
        """

        """
        with open(path, 'r', encoding=self.encoding) as f:
            return f.read()
