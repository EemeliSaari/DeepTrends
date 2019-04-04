import io
import logging
import os
import sys
import subprocess
import hashlib

sys.path.append('..')

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from utils.coroutines import run_coroutines
from base import BaseParser


class PDF2TextInterface(BaseParser):

    def __init__(self, path):
        super(PDF2TextInterface, self).__init__(path=path)

        self._check_availability()

    def __enter__(self):
        open(self.tempfile, mode='a').close()
        return super(PDF2TextInterface, self).__enter__()

    def __exit__(self, *args):
        if os.path.exists(self.tempfile):
            os.remove(self.tempfile)
        super(PDF2TextInterface, self).__exit__(*args)

    def parse(self):
        subprocess.call(['pdftotext', self.path, self.tempfile])
        with open(self.tempfile, encoding='utf-8') as f:
            text = f.read()
        return text

    def _check_availability(self):
        pass

    @property
    def tempfile(self):
        m = hashlib.sha256()
        m.update(bytes(self.path, 'utf-8'))
        return m.hexdigest() + '.txt'


class PDFMinerInterface(BaseParser):

    def __init__(self, path, codec='utf-8'):
        super(PDFMinerInterface, self).__init__(path=path)
        self.codec = codec

    def parse(self):
        retstr = io.StringIO()

        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, retstr, codec=self.codec, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        string = ''
        with open(self.path, 'rb') as f:
            for page in PDFPage.get_pages(f):
                interpreter.process_page(page)
                string += retstr.getvalue()

        return string


def pdf2txt(document, outfile, enforce=False, engine : str='pdf2text'):
    """PDF to TXT parser

    Attempts to parse the given pdf document into a text file

    Parameters
    ----------
    document : str, path-like
        Path to the document
    outfile : str, path-like
        Path to the output file
    engine : str
        Engine to be used for pdf parsing.
    """
    if os.path.exists(outfile) and not enforce:
        return

    ENGINE_MAP = dict(
        pdf2text=PDF2TextInterface, 
        pdfminer=PDFMinerInterface
    )

    with ENGINE_MAP[engine](path=document) as parser:
        #try:
        text = parser()
        #except Exception as e:
        #    logging.warn(f'Could not parse the file: {document} with error {str(e)}')
        #    return

    with open(outfile, 'w', encoding='utf-8') as out:
        out.write(text)


def parse_all_pdf(path : str, outpath : str, engine : str):
    """

    """
    async def routine(document, output):
        nonlocal engine
        pdf2txt(document, output, engine=engine)

    print(path)

    if not os.path.exists(path):
        raise OSError('Path {} does not exist.'.format(path))

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for p in os.listdir(path):
        folder = os.path.join(path, p)
        if os.path.isdir(folder):
            parse_all_pdf(folder, outpath=os.path.join(outpath, p), engine=engine)

    pdfs = list(filter(lambda p: '.pdf' in p, os.listdir(path)))

    if not pdfs:
        return

    inputs = map(lambda f: os.path.abspath(os.path.join(path, f)), pdfs)
    outputs = map(lambda f: os.path.join(outpath, f.replace('.pdf', '.txt')), pdfs)
    coroutines = [routine(*params) for params in zip(inputs, outputs)]

    run_coroutines(*coroutines)


if __name__ == '__main__':
    parse_all_pdf('../../data/papers/', '../../data/parsed', 'pdf2text')