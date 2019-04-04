import os

import requests


async def get_link_content(link : str, path : str, index : int = 1):
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

    if index % 50 == 0:
        print(f'Downloading file: {filename}')

    if os.path.exists(filepath):
        return

    with requests.get(url=link, stream=True) as res:
        with open(filepath, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
