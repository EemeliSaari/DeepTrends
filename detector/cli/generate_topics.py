import logging
import sys
import os

import click

sys.path.append('..')

from corpora import Corpora
from pipeline.embeddings import Word2VecWrapper
from pipeline.topics import (LDAWrapper, SHDPWrapper, corpus_topics,
                             get_model_words)




@click.command()
@click.option('--model', type=str)
@click.option('--n_topics', type=int)
@click.option('--data_path', type=str)
@click.option('--data_prefix', type=str)
@click.option('--result_path', type=str)
@click.option('--dictionary_path', type=str, default='')
@click.option('--vectors_path', type=str)
@click.option('--batch_size', type=int)
@click.option('--iterations', type=int)
@click.option('--passes', type=int)
@click.option('--shuffle')
def main(model, n_topics, data_path, data_prefix, result_path, dictionary_path, 
         vectors_path, batch_size, iterations, passes, shuffle):

    corpora_params = dict(
        data_path=data_path,
        prefix=data_prefix,
        dictionary=dictionary_path,
        iterator='bow'
    )

    if os.path.exists(dictionary_path):
        corpora = Corpora(**corpora_params)
    else:
        corpora = Corpora(**corpora_params).build()
        corpora.dictionary.save_as_text(dictionary_path)

    MAP = dict(
        lda=(LDAWrapper,
            dict(
                n_topics=n_topics,
                iterations=iterations,
                passes=passes,
                batch_size=batch_size,
                id2word=corpora.dictionary
            )
        ),
        shdp=(SHDPWrapper,
            dict(
                n_topics=n_topics,
                passes=passes,
                batch_size=batch_size,
                batch_shuffle=shuffle
            )
        )
    )

    model_class, params = MAP[model]
    topic_model = model_class(**params)
    topic_model.fit(corpora)

    if hasattr(topic_model, 'save'):
        topic_model.save()


def go():
    main()


if __name__ == '__main__':
    go()
