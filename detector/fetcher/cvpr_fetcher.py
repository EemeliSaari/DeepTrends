# CVPR Open Access parser - Papers from 2013-2018

import os
import random
import sys

import bs4 as bs
import requests

sys.path.append('..')

from base import BaseFetcher

BASE_URL = 'http://openaccess.thecvf.com/'


class CVPRFetcher(BaseFetcher):
    """

    """
    url = 'http://openaccess.thecvf.com/'
    prefix = 'CVPR{}'
    latest = 2018

    def __init__(self, save_dir : bool=True):
        super(CVPRFetcher, self).__init__(save_dir=save_dir)

    def fetch(self, year : int, output_path : str):
        """Fetch API for CVPR

        Download the published papers for given year in pdf format.

        Parameters
        ----------
        year : int, 2013 <= year <= 2018
            Year of the conference specified. 
        output_path : string, path-like
            Path to save files into.
        """
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        save_path = self._create_output(output_path, year)

        res = self.make_request(url=self.url+self.prefix.format(year)+'.py')

        soup = bs.BeautifulSoup(res.text, 'lxml')
        links = [BASE_URL+x['href'] for x in soup.findAll('a', href=True) if 'pdf' in x]

        if not links:
            raise ValueError(f'Did not find any pdf documents for year {year}')

        self.fetch_link_content(path=save_path, *links)

    @property
    def years(self):
        return list(range(2013, self.latest+1))
