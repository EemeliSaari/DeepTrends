import logging
import os
import sys
import datetime

import click

sys.path.append('..')
sys.path.append('../sHDP/')

print(sys.path)

from corpora import Corpora
from pipeline.embeddings import Word2VecWrapper, load_vectors
from pipeline.topics import (LDAWrapper, SHDPWrapper, document_topics,
                             model_words)


@click.command()
@click.option('--model', type=str)
@click.option('--alpha', type=int, default=1)
@click.option('--n_topics', type=int, default=50)
@click.option('--data_path', type=str)
@click.option('--data_prefix', type=str)
@click.option('--result_path', type=str, default='')
@click.option('--dictionary_path', type=str, default='')
@click.option('--vectors_path', type=str)
@click.option('--batch_size', type=int, default=10)
@click.option('--iterations', type=int, default=500)
@click.option('--passes', type=int, default=1)
@click.option('--n_words', type=int, default=15)
@click.option('--shuffle', '-s', is_flag=True)
def main(model, alpha, n_topics, data_path, data_prefix, result_path, dictionary_path, 
         vectors_path, batch_size, iterations, passes, n_words, shuffle):

    if not os.path.exists(result_path):
        raise OSError(f'Provided path {result_path} does not exist.')

    corpora_params = dict(
        data_path=data_path,
        prefix=data_prefix,
        iterator='bow'
    )

    if os.path.exists(dictionary_path):
        corpora = Corpora(dictionary=dictionary_path, **corpora_params)
    else:
        corpora = Corpora(**corpora_params).build()
        corpora.dictionary.save_as_text(dictionary_path)

    if len(corpora) == 0:
        raise ValueError(f'Did not find any documents from path: {data_path} for given prefix {data_prefix}')


    MAP = dict(
        lda=(LDAWrapper,
            dict(
                n_topics=n_topics,
                alpha=alpha,
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
                batch_shuffle=shuffle,
                vector_map=load_vectors(vectors_path, dictionary=corpora.dictionary),
                num_docs=len(corpora)
            )
        )
    )

    model_class, params = MAP[model]
    topic_model = model_class(**params)
    topic_model.fit(corpora)

    model_name = str(topic_model) + f'_{data_prefix}'
    years = corpora.years
    if years:
        model_name += f'{years[0]}-{years[-1]}'
    if shuffle:
        model_name += f'_shuffled'

    path_dir = os.path.join(result_path, model_name)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    if hasattr(topic_model, 'save'):
        topic_model.save(model_name)

    topic_df = document_topics(topic_model, corpora)
    
    dictionary = None
    if model == 'shdp':
        dictionary = corpora.dictionary

    words_df = model_words(topic_model, n=n_words, dictionary=dictionary)

    topics_path = os.path.join(path_dir, 'topics_' + model_name + '.csv')
    words_path = os.path.join(path_dir, 'words_' + model_name + '.csv')

    print(topic_df.head())
    print(topic_df.tail())
    print(topic_df['year'].unique())

    topic_df.to_csv(topics_path, index=False)
    words_df.to_csv(words_path, index=False)


def go():
    main()


if __name__ == '__main__':
    go()
