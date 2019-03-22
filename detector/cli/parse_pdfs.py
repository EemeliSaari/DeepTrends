import click

import logging
import sys

sys.path.append('..')

from pipeline.parser import parse_all_pdf


@click.command()
@click.option('--input_path', help='Input directory as path.')
@click.option('--output_path', help='Output directory to save results into.')
@click.option('--engine', type=str, default='pdfminer', help='Engine to be used for parsing process')
def main(input_path, output_path, engine):
    logging.basicConfig(level=logging.WARN)

    parse_all_pdf(path=input_path, outpath=output_path, engine=engine)


def go():
    main()


if __name__ == '__main__':
    go()
