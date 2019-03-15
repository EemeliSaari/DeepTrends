import collections
import logging
import os
import time
import re

from filelock import FileLock
from gensim.parsing import preprocess_string
from pipeline.parser import custom_preprocess


class Corpora:

    PARSERS = dict(gensim=preprocess_string, simple=custom_preprocess)
    n_docs = 0
    is_loaded = False

    def __init__(self, path, parsing='gensim', encoding='utf-8', keep_tokens=True):
        self.path = path
        self.encoding = encoding
        self.keep_tokens = keep_tokens

        self.__parsing_method = parsing
        self.__tokens = []
        self.__raw = []

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(loaded={self.is_loaded}, n_documents={self.n_docs})'

    def load(self):
        if os.path.isdir(self.path):
            self.__raw = list(self._read_folder())
        else:
            with FileLock(self.path):
                self.__raw = [self._read_file(self.path)]
        self.n_docs = len(self.__raw)
        self.is_loaded = True

        return self

    def clear(self):
        if self.is_loaded:
            self.n_docs = 0
            self.__raw = []

        if not self.keep_tokens:
            self.__tokens = []

    @property
    def raw(self):
        return self.__raw

    @property
    def tokens(self):
        start = time.time()
        self.__tokens = list(map(self.PARSERS[self.__parsing_method], self.raw))
        logging.info('Corpora tokens took: {:.5f}s'.format(time.time()-start))
        return self.__tokens

    @property
    def id2doc(self):
        return self.__documents

    @property
    def year(self):
        return re.findall('\d+', self.path)[-1]

    def _read_folder(self):
        self.__documents = os.listdir(self.path)
        return map(lambda f: self._read_file(os.path.join(self.path, f)), self.__documents)

    def _read_file(self, path):
        with open(path, 'r', encoding=self.encoding) as f:
            return ''.join(f.readlines())
