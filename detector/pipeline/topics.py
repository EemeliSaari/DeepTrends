import itertools
import logging

import numpy as np
import pandas as pd
from gensim.models.ldamulticore import LdaModel

from base import BaseModel
#from utils.namings import topic_columns

try:
    from HDP.models import HDP
    from HDP.util.general import sgd_passes
    from HDP.util.text import progprint
    from core.core_distributions import vonMisesFisherLogNormal as vMF
except ImportError:
    logging.warn('Could not import modules for sHDP')


class LDAWrapper(BaseModel):
    """

    """
    obj = LdaModel

    def corpus_to_topics(self, corpus):
        """

        """
        return np.array(self.obj.get_document_topics(corpus, minimum_probability=0))[:, :, 1]

    def topic_words(self, n_words=10):
        """

        """
        parse_word = lambda w: w.split('*')[1].strip().replace('"', '')
        words = [[] for i in range(self.obj.num_topics)]
        for topic_id, text in self.obj.show_topics(num_topics=self.obj.num_topics, num_words=n_words):
            topic_words = [parse_word(x) for x in text.split('+')]
            words[int(topic_id)] = topic_words
        return np.array(words)


class SHDPWrapper(BaseModel):
    """

    """

    obj = HDP

    def __init__(self, n_topics : int=100, dim : int=100, alpha : int=1, 
                 gamma : int=2, sigma_0 : float=0.25, tau : float=0.8,
                 C_0 : int=1, m_0 : int=2, kappa_sgd : float=0.6, 
                 batch_size : int=10, n_passes : int=1, num_docs : int=None, 
                 seed : int=42, **kwargs):
        self.__n_topics = n_topics
        self.__dim = dim
        self.__alpha = alpha
        self.__gamma = gamma
        self.__sigma_0 = sigma_0
        self.__tau = tau
        self.__C_0 = C_0
        self.__m_0 = m_0
        self.__kappa_sgd = kappa_sgd
        self.__batch_size = batch_size
        self.__n_passes = n_passes
        self.__num_docs = num_docs
        self.__seed = seed

        self._initialize_components()

        self.validate_parameters()

        super(SHDPWrapper, self).__init__(
            alpha=self.__alpha, 
            gamma=self.__gamma,
            obs_distns=self.__components_0,
            num_docs=num_docs,
            **kwargs)

    def validate_parameters(self):
        if not 0.5 < self.__kappa_sgd <= 1:
            raise ValueError(f'Parameter kappa_sgd {self.__kappa_sgd} is invalid')
        
        if not self.__tau >= 0:
            raise ValueError(f'Parameter tau {self.__tau} is invalid.')

    def fit(self, X, y=None):
        """

        """
        for doc, N in X:
            for data, rho_t in self._sgd_passes(doc):
                self.obj.meanfield_sgdstep(
                    minibatch=data, 
                    minibatchfrac=np.array(data).shape[0] / N,
                    stepsize=rho_t
                )

        self.fitted_ = True

        return self

    def transform(self, X):

        self.__doc_states = np.empty((len(X), self.__n_topics))
        for i in range(self.__doc_states.shape[0]):
            self.obj.add_data(np.atleast_2d(X[i][0].squeeze()), i)
            self.obj.states_list[-1].meanfieldupdate()
            self.__doc_states[:, :] = self.obj.states_list[-1].all_expected_stats[0] 

        return self.tokens()

    def topics(self):
        """

        """

        topic_dist = np.empty((len(X), hdp.num_states))
        for i in range(self.__doc_states.shape[0]):
            topic_dist[i, :] = np.average(self.__doc_states[i, :], 0)
        return topic_dist

    def topic_words(self, n_words=15):
        """

        """
        topics_dict = {}
        for i in range(hdp.num_states):
            topics_dict[i] = collections.defaultdict(float)

        #TODO: Track the words!
        for doc in self.__words:
            for idx, word_id in enumerate(doc):
                for t in range(K):
                    topics_dict[t][word_id] += temp_exp[idx, t]

        sorted_topic_words = []
        for t in range(K):
            top_words = [dictionary[k] for k in sorted(topics_dict[t], key=lambda x: topics_dict[t][x], reverse=True)[:n_words]]
            sorted_topic_words.append(top_words)

        return np.array(sorted_topic_words)

    def _initialize_components(self):
        """

        """
        np.random.seed(self.__seed)

        d = np.random.rand(self.__dim,)
        d = d/np.linalg.norm(d)

        obs_hypparams = dict(
            mu_0=d, C_0=self.__C_0, m_0=self.__m_0, sigma_0=self.__sigma_0
        )
        self.__components_0 = [vMF(**obs_hypparams) for _ in range(self.__n_topics)]

    def _sgd_steps(self):
        """

        """
        for c in itertools.count(1):
            yield (c + self.__tau)**(-self.__kappa_sgd)

    def _sgd_passes(self, X):
        """

        """
        N = len(X)
        print(N)

        for itr in range(self.__n_passes):
            perm = np.random.permutation(N)

            if self.__batch_size == 1:
                for idx, rho_t in zip(perm,self._sgd_steps()):
                    yield X[idx], rho_t
            else:
                minibatch_indices = np.array_split(perm, N/self.__batch_size)
                for indices, rho_t in zip(minibatch_indices, self._sgd_steps()):
                    yield [X[idx] for idx in indices], rho_t


def corpus_topics(topics, corpus):
    """

    """
    topic_df = pd.DataFrame(topics, columns=topic_columns(n=topics.shape[1]))

    meta = [(corpus.year, corpus.id2doc[i]) for i in range(topics.shape[0])]
    meta_df = pd.DataFrame(meta, columns=['year', 'doc'])

    return topic_df.merge(meta_df, how='outer', right_index=True, left_index=True)


def get_model_words(topic_model, n=10):
    """

    """
    words = np.array(topic_model.topic_words(n_words=n))

    df = pd.DataFrame(words.T, columns=topic_columns(n=topic_model.num_topics))

    return df
