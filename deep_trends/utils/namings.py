

def topic_columns(n, prefix='Topic'):
    """

    """
    topic_name = lambda x: prefix + f'-{x}'
    return list(map(topic_name, range(n)))
