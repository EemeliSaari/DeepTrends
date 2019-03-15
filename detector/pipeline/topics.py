import numpy as np
import pandas as pd
from gensim.models.ldamulticore import LdaModel

from base import BaseModel
from utils.namings import topic_columns


class LDAWrapper(BaseModel):
    """

    """
    obj = LdaModel

    def __init__(self, *args, **kwargs):
        super(LDAWrapper, self).__init__(*args, **kwargs)

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
