import os
import re
import sys

import bs4 as bs
import requests

sys.path.append('..')

from base import BaseFetcher
from fetcher.processes import get_link_content
from utils.coroutines import run_coroutines


BASE_URL = 'http://papers.nips.cc/'


class NIPSFetcher(BaseFetcher):
    """

    """
    url = 'http://papers.nips.cc/'
    prefix = 'NIPS{}'

    def __init__(self, save_dir : bool=True):
        super(NIPSFetcher, self).__init__(self, save_dir=save_dir)

        self.__year_map = self.find_years()

    def fetch(self, year : int, output_path : str):
        """

        """
        save_path = self._create_output(output_path, year)

        res = self.make_request(url=self.url + self.__year_map[year])
        soup = bs.BeautifulSoup(res.text, 'lxml')

        paper_list = sorted(soup.findAll('ul'), key=lambda x: len(x), reverse=True)[0]

        paper_links = list(filter(lambda l: l.split('/')[1] == 'paper', self.find_links(paper_list)))
        links = list(map(lambda l: self.url + l + '.pdf', paper_links))

        coroutines = [get_link_content(l, save_path, index=i) for i, l in enumerate(links)]
        _ = run_coroutines(*coroutines)

    def find_years(self):
        """

        """
        res = self.make_request(url=self.url)

        parse_year = lambda x: (lambda y: y[0] if y else None)(re.findall(r'NIPS\s([0-9]*)', x))
        get_links = lambda x: (parse_year(x.text), x['href'])

        soup = bs.BeautifulSoup(res.text, 'lxml')

        raw = [get_links(x.find('a', href=True)) for x in soup.findAll('li')]
        years = dict((int(k), v) for k, v in raw if k)

        return years

    @property
    def years(self):
        return list(self.__year_map.keys())
