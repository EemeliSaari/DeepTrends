import asyncio


def loop_wrapper(func):
    """

    """
    def wrapper(*args):
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(func(*args))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
        return results
    return wrapper


@loop_wrapper
async def run_coroutines(*args):
    """Coroutine runner

    Waits for given coroutines to finish.

    Parameters
    ----------
    args : tuple of coroutines
        List of coroutines to be run.
    """
    return await asyncio.gather(*args)
