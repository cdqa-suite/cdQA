import json
import os
from tqdm import tqdm
import uuid

def df2squad(df, filename, squad_version='v2.0', output_dir=None):
   """
    Converts a pandas dataframe with columns ['title', 'content'] to a json file with SQuAD format.

    Parameters
   ----------
    df : pandas.DataFrame
        a pandas dataframe with columns ['title', 'content']
    filename : [type]
        [description]
    squad_version : str, optional
        the SQuAD dataset version format (the default is 'v2.0')
    output_dir : [type], optional
        Enable export of output (the default is None)
   
   Returns
   -------
   json_data
       A json object with SQuAD format

    Examples
    --------
    >>> 

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

def generate_squad_examples(question, article_indices, metadata):
    """
    [summary]
    
    Parameters
    ----------
    question : [type]
        [description]
    article_indices : [type]
        [description]
    metadata : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]

    Examples
    --------
    >>> 

    """

    
    squad_examples = []
    
    metadata_sliced = metadata.loc[article_indices]
    
    for index, row in tqdm(metadata_sliced.iterrows()):
        temp = {'title': row['title'],
               'paragraphs': []}
        
        for paragraph in row['paragraphs']:
            temp['paragraphs'].append({'context': paragraph,
                                       'qas': [{'answers': [],
                                                'question': question,
                                                'id': str(uuid.uuid1())}]
                                      })

            squad_examples.append(temp)

    return squad_examples

def filter_paragraphs(paragraphs):
    """
    [summary]
    
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
    >>> 

    """

    # filter out paragraphs shorter than 10 words and longer than 250 words
    paragraphs_filtered = [paragraph for paragraph in paragraphs if len(paragraph.split()) >= 10 and len(paragraph.split()) <= 250]
    return paragraphs_filtered
