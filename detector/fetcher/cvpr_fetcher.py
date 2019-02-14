# CVPR Open Access parser - Papers from 2013-2018

import os
import random

import requests
import bs4 as bs

from utils.coroutines import run_coroutines


BASE_URL = 'http://openaccess.thecvf.com/'
LATEST_YEAR = 2018


def fetch_all(output_path : str):
    """All Open access papers for CVPR

    Small wrapper to fetch all available papers

    Parameters
    ----------
    output_path : string, path-like
        Path to save files into.
    """
    for year in range(2013, LATEST_YEAR + 1, 1):
        fetch(year=year, output_path=output_path)


def fetch(year : int, output_path : str, save_dir : bool = True, sample_size : int = 0):
    """Fetch API for CVPR

    Download the published papers for given year in pdf format.

    Parameters
    ----------
    year : int, 2013 <= year <= 2018
        Year of the conference specified. 
    output_path : string, path-like
        Path to save files into.
    save_dir : bool, (optional) 
        Wheter or not save the papers under directory.
    sample_size : int, (optional), default=0
        Sample size to be used.
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    session_name = 'CVPR{:d}'.format(year)

    dir_path = output_path
    if save_dir:
        dir_path = os.path.join(output_path, session_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    url = BASE_URL+session_name+'.py'
    res = requests.get(url=url)
    if res.status_code is not 200:
        raise ValueError('Fetch failed with status code: {:d}'.format(res.status_code))

    soup = bs.BeautifulSoup(res.text, 'lxml')
    links = [BASE_URL+x['href'] for x in soup.findAll('a', href=True) if 'pdf' in x]

    if not links:
        raise ValueError('Did not find any pdf documents from {:s}'.format(url))

    if len(links) > sample_size > 0:
        links = random.sample(links, sample_size)

    coroutines = [get_link_content(l, dir_path) for l in links]
    results = run_coroutines(*coroutines)
    #TODO: Check and rerun failed ones.


async def get_link_content(link : str, path : str):
    """Async link content fetcher.

    Downloads the link content and saves it to the given path.

    Parameters
    ----------
    link : str, url-like
        URL for the content to be downloaded
    path : str, path-like
        Path for the content to be saved into.
    """
    filename = link.split('/')[-1]
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        return

    with requests.get(url=link, stream=True) as res:
        with open(filepath, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
