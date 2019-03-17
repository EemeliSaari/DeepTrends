from gensim import parsing


def custom_preprocess(doc):
    """

    """
    as_lower = lambda doc: doc.lower()
    filters = [
        parsing.strip_tags,
        parsing.strip_punctuation,
        parsing.strip_multiple_whitespaces,
        parsing.strip_numeric,
        as_lower,
        parsing.remove_stopwords,
    ]
    for f in filters:
        doc = f(doc)
    doc = parsing.strip_short(doc, minsize=2)
    return doc.split(' ')
