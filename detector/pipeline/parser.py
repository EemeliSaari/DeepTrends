import io
import sys

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


def pdf2txt(document, outfile):
    """PDF to TXT parser

    Attempts to parse the given pdf document into a text file

    Parameters
    ----------
    document : str, path-like
        Path to the document
    outfile : str, path-like
        Path to the output file
    """
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with open(document, 'rb') as f:
        with open(outfile, 'w', encoding='utf-8') as out:
            for page in PDFPage.get_pages(f):
                interpreter.process_page(page)
                out.write(retstr.getvalue())
