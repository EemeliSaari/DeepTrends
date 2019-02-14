import asyncio


def run_coroutines(*args):

    loop = asyncio.new_event_loop()
    tasks = [loop.create_task(cr) for cr in args]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    return tasks
