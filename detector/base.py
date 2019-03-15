import logging
import os

import requests
from filelock import FileLock

from fetcher.processes import get_link_content
from utils.coroutines import run_coroutines


class _Base:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logging.error(f'{str(exc_type)} from {self.__class__.__name__} with msg: {exc_val}')


class BaseParser(_Base):
    def __init__(self, path):
        if not os.path.exists(path):
            raise AssertionError(f'Path {path} does not exists.')
        self.path = path

    def __call__(self):
        return self.parse()

    def parse(self):
        raise NotImplementedError()


class BaseFetcher(_Base):
    """

    """
    url = None
    prefix = '{}'

    def __init__(self, save_dir : bool=True):
        self.save_dir = save_dir

    def find_links(self, tags):
        """

        """
        return [x['href'] for x in tags.findAll('a', href=True)]

    def make_request(self, url):
        """

        """
        res = requests.get(url=url)
        if res.status_code is not 200:
            raise ValueError(f'Fetch failed with status code: {res.status_code}')
        return res

    def fetch_link_content(self, path, *links):
        """

        """
        coroutines = [get_link_content(l, path) for l in links]
        _ = run_coroutines(*coroutines)

    def _create_output(self, output_path : str, year : int):
        """

        """
        dir_path = output_path
        if self.save_dir:
            dir_path = os.path.join(output_path, self.prefix.format(year))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        return dir_path


class BaseModel(_Base):
    """

    """
    obj = None

    def __init__(self, *args, **kwargs):
        self.obj = self.obj(*args, **kwargs)

    def __getattr__(self, attr):
        """Overrided getattr method

        """
        if hasattr(super(BaseModel, self), attr):
            return super(BaseModel, self).__getattr__(attr)
        elif hasattr(self.obj, attr):
            value = getattr(self.obj, attr)
        else:
            raise AttributeError(f'Attribute {attr} not found!')
        
        if callable(value):
            def hook(*args, **kwargs):
                """

                """
                result = value(*args, **kwargs)
                if isinstance(result, type(self.obj)):
                    return self
                return result
            return hook
        else:
            return value
