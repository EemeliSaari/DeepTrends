from collections import defaultdict
from itertools import chain

from gensim.topic_coherence import segmentation, direct_confirmation_measure, text_analysis

def inverted_accumulator(corpus, relevant_words):
    """

    """
    inverted_index = defaultdict(set)
    id2contiguous = {word_id: i for i, word_id in enumerate(relevant_words)}
    for i, (t, _) in enumerate(corpus):
        filtered = list(filter(lambda w: w in relevant_words, corpus.dictionary.doc2idx(t)))
        for idx in filtered:
            inverted_index[id2contiguous[idx]].add(i)

    accumulator = text_analysis.InvertedIndexAccumulator(relevant_words, dictionary=corpus.dictionary)
    accumulator._inverted_index = inverted_index
    accumulator._num_docs = len(corpus)
    
    return accumulator


def pmi(topics, corpus):
    """

    """
    unique_ids = unique_ids = set(chain(*topics))

    accumulator = inverted_accumulator(corpus, unique_ids)

    segments = segmentation.s_one_pre(topics)

    scores = direct_confirmation_measure.log_ratio_measure(segments, accumulator)

    return np.array(scores)
