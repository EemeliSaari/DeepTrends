from fetcher.nips_fetcher import NIPSFetcher
from fetcher.cvpr_fetcher import CVPRFetcher


DATASET_MAP = dict(
    nips=NIPSFetcher,
    cvpr=CVPRFetcher
)


def download(dataset : str, output : str):
    """Downloader

    Small wrapper to fetch all available papers

    Parameters
    ----------
    output_path : string, path-like
        Path to save files into.
    """
    fetcher = DATASET_MAP[dataset]

    for year in fetcher.years:
        fetcher.fetch(year=year, output=output)
