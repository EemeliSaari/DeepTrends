import os
import sys

import click
import pandas as pd
from gensim.corpora.dictionary import Dictionary

sys.path.append('..')

from pipeline.embeddings import Word2VecWrapper
from corpora import Corpora

def initialize_corpora(data_path, data_prefix, dict_path, iterator, **kwargs):

    corpora_params = dict(
        data_path=data_path,
        prefix=data_prefix
    )

    if os.path.exists(dict_path):
        corpora = Corpora(dictionary=dict_path, **corpora_params)
    else:
        corpora = Corpora(**corpora_params).build()
        corpora.dictionary.save_as_text(dict_path)

    if len(corpora) == 0:
        raise ValueError(f'Did not find any documents from path: {data_path} for given prefix {data_prefix}')

    return corpora


@click.command()
@click.option('--model', type=str)
@click.option('--weights', type=str)
@click.option('--data_path', type=str)
@click.option('--data_prefix', type=str)
@click.option('--result_path', type=str)
@click.option('--dictionary_path', type=str)
@click.option('--size', type=int, default=100)
@click.option('--window', type=int, default=5)
@click.option('--min_count', type=int)
@click.option('--batch_size', type=int)
@click.option('--epochs', type=int, default=50)
@click.option('--normalize', type=str)
def main(model, weights, data_path, data_prefix, result_path, dictionary_path, size, window,
         min_count, batch_size, epochs, normalize):

    dictionary = None
    if data_path:
        corpora = initialize_corpora(data_path, data_prefix, dictionary_path, 
            'token')
        dictionary = corpora.dictionary
    elif not data_path and dictionary_path:
        dictionary = Dictionary.load_from_text(dictionary_path)

    MAP = dict(
        word2vec=(Word2VecWrapper,
            dict(
                weights=weights,
                size=size,
                window=window,
                min_count=min_count,
                normalize=normalize,
                dictionary=dictionary,
                batch_size=batch_size
            )
        ),
    )

    model_class, params = MAP[model]
    model = model_class(**params)

    # Not training
    if weights and not data_path:
        vector_dict = model.vectors
    elif data_path and not weights:
        model.fit(corpora, epochs=epochs)
        vector_dict = model.transform(corpora)
    else:
        raise ValueError('Need to define ether data_path or weights.')

    vectors_path = os.path.join(result_path, str(model)+'.csv')
    pd.DataFrame(vector_dict).to_csv(vectors_path, index=False)

    print(f'Vectors stored to path: {vectors_path}')


def go():
    main()


if __name__ == '__main__':
    go()
