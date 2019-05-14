def filter_paragraphs(paragraphs, min_length=10, max_length=250):
    """
    Filters out paragraphs shorter than X words and longer than Y words

    Parameters
    ----------
    paragraphs : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Examples
    --------
    >>> from ast import literal_eval
    >>> import pandas as pd
    >>> from cdqa.utils.filters import filter_paragraphs

    >>> df = pd.read_csv('../data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
    >>> df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)

    """

    paragraphs_filtered = [paragraph for paragraph in paragraphs if len(
        paragraph.split()) >= min_length and len(paragraph.split()) <= max_length]
    return paragraphs_filtered
