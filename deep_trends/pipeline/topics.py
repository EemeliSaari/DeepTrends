import collections
import gc
import itertools
import logging

import mock
import numpy as np
import pandas as pd
from gensim.models.ldamulticore import LdaModel

from base import BaseModel
from mocks.shdp_mocks import vmf_init
from utils.namings import topic_columns

try:
    from HDP.models import HDP
    from core.core_distributions import vonMisesFisherLogNormal as vMF
except ImportError as e:
    logging.warn(f'Could not import modules for sHDP: {str(e)}')


class LDAWrapper(BaseModel):
    """

    """
    obj = LdaModel

    def __init__(self, n_topics : int=100, alpha='auto', iterations : int=500, 
                 update_every : int=0, passes : int=100, eval_every=None, 
                 batch_size : int=2000, verbose : bool=False,
                 **kwargs):
        self.__chunksize = batch_size
        self.__passes = passes
        self.__iterations = iterations
        self.__eval_every = eval_every
        self.__update_every = update_every
        self.__alpha = alpha

        self.verbose = verbose

        super(LDAWrapper, self).__init__(
            alpha=alpha,
            eval_every=eval_every,
            num_topics=n_topics,
            update_every=self.__update_every,
            **kwargs)

    def __str__(self):
        return f'lda_{self.obj.num_topics}K_{self.__alpha}alpha_{self.__iterations}iters_{self.__passes}passes'

    def fit(self, X, y=None):

        self.obj.update(
            corpus=X,
            passes=self.__passes,
            chunksize=self.__chunksize,
            iterations=self.__iterations
        )

        self.fitted_ = True

        return self

    def transform(self, X):
        """

        """
        docs = [d for d, N in X]
        topics = self.obj.get_document_topics(docs, minimum_probability=0)
        return np.array(topics)[:, :, 1]

    def topic_words(self, n_words=10):
        """

        """
        parse_word = lambda w: w.split('*')[1].strip().replace('"', '')
        words = [[] for i in range(self.obj.num_topics)]
        for topic_id, text in self.obj.show_topics(num_topics=self.obj.num_topics, num_words=n_words):
            topic_words = [parse_word(x) for x in text.split('+')]
            words[int(topic_id)] = topic_words
        return words


@mock.patch('sHDP.core.core_distributions.vonMisesFisherLogNormal.__init__', vmf_init)
class SHDPWrapper(BaseModel):
    """

    """

    def __init__(self, n_topics : int=100, dim : int=None, alpha : int=1, 
                 gamma : int=2, sigma_0 : float=0.25, tau : float=0.8,
                 C_0 : int=1, m_0 : int=2, kappa_sgd : float=0.6, 
                 batch_size : int=10, passes : int=1, num_docs : int=None, 
                 seed : int=42, vector_map=None, verbose : bool=False,
                 batch_shuffle : bool = True, **kwargs):
        self.obj = HDP
        self.__n_topics = n_topics
        if isinstance(alpha, str):
            self.__alpha = float(alpha)
        else:
            self.__alpha = alpha
        self.__gamma = gamma
        self.__sigma_0 = sigma_0
        self.__tau = tau
        self.__C_0 = C_0
        self.__m_0 = m_0
        self.__kappa_sgd = kappa_sgd
        self.__num_docs = num_docs
        self.__seed = seed

        self.vector_map = vector_map

        if self.vector_map is None:
            raise AssertionError('Expecteed vector map to be provided.')
        if not dim:
            self.__dim = self.vector_map.shape[0]
        else:
            self.__dim = dim

        self.verbose = verbose
        self.batch_size = batch_size
        self.passes = passes
        self.batch_shuffle = batch_shuffle

        self.__words = [] # Keep the count for the upcoming words.
        self.__doc_states = []

        self._initialize_components()

        self.validate_parameters()

        super(SHDPWrapper, self).__init__(
            alpha=self.__alpha, 
            gamma=self.__gamma,
            obs_distns=self.__components_0,
            num_docs=num_docs+1,
            **kwargs)

    def __str__(self):
        return f'shdp_{self.__n_topics}K_{self.__alpha}alpha_{self.__gamma}gamma_{self.__dim}dim_{self.passes}passes_{self.__gamma}gamma'

    def validate_parameters(self):
        """

        """
        if not 0.5 < self.__kappa_sgd <= 1:
            raise ValueError(f'Parameter kappa_sgd {self.__kappa_sgd} is invalid')

        if not self.__tau >= 0:
            raise ValueError(f'Parameter tau {self.__tau} is invalid.')

        if not self.batch_size >= 1:
            raise ValueError(f'Batch size expected to be larger than 1, received {self.batch_size}')

    def fit(self, X, y=None):
        """

        """
        i = 0
        print('Starting the fitting process!')
        for (doc, N), rho_t in zip(self.glovize(X), self._sgd_steps()):
            print(f'Document {i}/{N}')
            for _ in range(self.passes):
                if self.batch_shuffle and len(doc) > 1:
                    np.random.shuffle(doc)
                self.obj.meanfield_sgdstep(
                    doc,
                    np.array(doc).shape[0] / np.float(N),
                    rho_t
                )
            i += len(doc)

        self.fitted_ = True

        return self

    def transform(self, X):
        """

        """
        self.__index = len(self.__doc_states)
        for i, (doc, _) in enumerate(self.glovize(X, skip_batch=True, update_words=True)):
            self.__doc_states.append(self._get_state(doc, i))
        self.obj._clear_caches()
        return self.topics()

    def topics(self):
        """

        """
        topic_dist = np.empty((len(self.__doc_states)-self.__index, self.obj.num_states))
        for i, idx in enumerate(range(self.__index, len(self.__doc_states))):
            topic_dist[i, :] = np.average(self.__doc_states[idx], 0)
        return topic_dist

    def topic_words(self, n_words=15):
        """

        """
        topics_dict = {}
        for i in range(self.obj.num_states):
            topics_dict[i] = collections.defaultdict(float)

        for i, doc in enumerate(self.__words):
            state = self.__doc_states[i]
            for idx, word_id in enumerate(doc):
                for t in range(self.obj.num_states):
                    topics_dict[t][word_id] += state[idx, t]

        sorted_topic_words = []
        for t in range(self.obj.num_states):
            top_words = [k for k in sorted(topics_dict[t], key=lambda x: topics_dict[t][x], reverse=True)[:n_words]]
            sorted_topic_words.append(top_words)

        return sorted_topic_words

    def glovize(self, bow, skip_batch=False, update_words=False):
        """

        """
        gc.collect()

        if not hasattr(self, 'vector_map'):
            raise AssertionError('Cant glovize without vector map.')

        N = len(bow)

        batch = []
        for i, doc_bow in enumerate(bow):
            if self.verbose and i % 10 == 0:
                print(f'Document {i+1}/{N}')
            if len(doc_bow) == 0 or len(doc_bow[0]) == 0:
                logging.warn(f'Received empty bow - skipping.')
                continue

            bow_vectors = []
            words = []
            missing = []
            for word, count in doc_bow:
                if word not in self.vector_map.columns:
                    missing.append(word)
                    continue
                vector = self.vector_map[word].values

                if vector.shape[0] == 0:
                    continue
                bow_vectors.append((vector, count))
                words.append(word)

            if update_words:
                self.__words.append(np.array(words))

            output = (np.array(bow_vectors), i)
            if len(output) > 0:
                if self.batch_size == 1 or skip_batch:
                    yield output, N
                    continue

                batch.append(output)
                if i == N - 1:
                    pass
                elif len(batch) < self.batch_size:
                    continue
            else:
                continue

            yield batch, N
            batch = []

    def _get_state(self, doc, index):
        """

        """
        self.obj.add_data(np.atleast_2d(doc[0].squeeze()), index)
        self.obj.states_list[-1].meanfieldupdate()
        return self.obj.states_list[-1].all_expected_stats[0]

    def _initialize_components(self):
        """

        """
        np.random.seed(self.__seed)

        d = np.random.rand(self.__dim)
        d = d/np.linalg.norm(d)

        obs_hypparams = dict(
            mu_0=d,
            C_0=self.__C_0,
            m_0=self.__m_0,
            sigma_0=self.__sigma_0
        )
        self.__components_0 = [vMF(**obs_hypparams) for _ in range(self.__n_topics)]

    def _sgd_steps(self):
        """

        """
        for c in itertools.count(1):
            yield (c + self.__tau)**(-self.__kappa_sgd)


def topics_df(topics, corpus):
    """

    """
    topic_df = pd.DataFrame(topics, columns=topic_columns(n=topics.shape[1]))
    meta = [(corpus.year, corpus.id2doc[i]) for i in range(topics.shape[0])]
    meta_df = pd.DataFrame(meta, columns=['year', 'doc'])

    return topic_df.merge(meta_df, how='outer', right_index=True, left_index=True)


def document_topics(model, corpora):
    """

    """
    dfs = []
    for c in corpora.corpus:
        bow = [d for d, _ in corpora[c.path]]
        try:
            topics = model.transform(bow)
        except MemoryError as e:
            print(c.path)
            raise e
        dfs.append(topics_df(topics, c))
    return pd.concat(dfs, axis=0)


def model_words(topic_model, n=10, dictionary=None):
    """

    """
    words = topic_model.topic_words(n_words=n)
    if dictionary:
        words = [[dictionary[w] for w in seq] for seq in words]

    words = np.array(words)

    n_topics = words.shape[0]

    df = pd.DataFrame(words.T, columns=topic_columns(n=n_topics))

    return df
