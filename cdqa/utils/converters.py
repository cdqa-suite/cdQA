import json
import os
from tqdm import tqdm
import uuid


def df2squad(df, squad_version='v1.1', output_dir=None, filename=None):
    """
     Converts a pandas dataframe with columns ['title', 'paragraphs'] to a json file with SQuAD format.

     Parameters
    ----------
     df : pandas.DataFrame
         a pandas dataframe with columns ['title', 'paragraphs']
     squad_version : str, optional
         the SQuAD dataset version format (the default is 'v2.0')
     output_dir : str, optional
         Enable export of output (the default is None)
     filename : str, optional
         [description]

    Returns
    -------
    json_data: dict
        A json object with SQuAD format

     Examples
     --------
     >>> from ast import literal_eval
     >>> import pandas as pd
     >>> from cdqa.utils.converter import df2squad, filter_paragraphs

     >>> df = pd.read_csv('../data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
     >>> df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)

     >>> json_data = df2squad(df=df, squad_version='v1.1', output_dir='../data', filename='bnpp_newsroom-v1.1')
    """

    json_data = {}
    json_data['version'] = squad_version
    json_data['data'] = []

    for index, row in tqdm(df.iterrows()):
        temp = {'title': row['title'],
                'paragraphs': []}
        for paragraph in row['paragraphs']:
            temp['paragraphs'].append({'context': paragraph,
                                       'qas': []})
        json_data['data'].append(temp)

    if output_dir:
        with open(os.path.join(output_dir, '{}.json'.format(filename)), 'w') as outfile:
            json.dump(json_data, outfile)

    return json_data


def generate_squad_examples(question, closest_docs_indices, metadata):
    """
    Creates a SQuAD examples json object for a given for a given question using outputs of retriever and document database.

    Parameters
    ----------
    question : [type]
        [description]
    closest_docs_indices : [type]
        [description]
    metadata : [type]
        [description]

    Returns
    -------
    squad_examples: list
        [description]

    Examples
    --------
    >>> from cdqa.utils.converter import generate_squad_examples
    >>> squad_examples = generate_squad_examples(question='Since when does the the Excellence Program of BNP Paribas exist?',
                                         closest_docs_indices=[788, 408, 2419],
                                         metadata=df)

    """

    squad_examples = []

    metadata_sliced = metadata.loc[closest_docs_indices]

    for index, row in tqdm(metadata_sliced.iterrows()):
        temp = {'title': row['title'],
                'paragraphs': []}

        for paragraph in row['paragraphs']:
            temp['paragraphs'].append({'context': paragraph,
                                       'qas': [{'answers': [],
                                                'question': question,
                                                'id': str(uuid.uuid4())}]
                                       })

        squad_examples.append(temp)

    return squad_examples
