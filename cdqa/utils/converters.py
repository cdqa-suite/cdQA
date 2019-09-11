import json
import os
import re
import sys
from tqdm import tqdm
from tika import parser
import pandas as pd
import uuid
import markdown
from pathlib import Path
from html.parser import HTMLParser


def df2squad(df, squad_version="v1.1", output_dir=None, filename=None):
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
    json_data["version"] = squad_version
    json_data["data"] = []

    for index, row in tqdm(df.iterrows()):
        temp = {"title": row["title"], "paragraphs": []}
        for paragraph in row["paragraphs"]:
            temp["paragraphs"].append({"context": paragraph, "qas": []})
        json_data["data"].append(temp)

    if output_dir:
        with open(os.path.join(output_dir, "{}.json".format(filename)), "w") as outfile:
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

    for index, row in metadata_sliced.iterrows():
        temp = {"title": row["title"], "paragraphs": []}

        for paragraph in row["paragraphs"]:
            temp["paragraphs"].append(
                {
                    "context": paragraph,
                    "qas": [
                        {"answers": [], "question": question, "id": str(uuid.uuid4())}
                    ],
                }
            )

        squad_examples.append(temp)

    return squad_examples


def pdf_converter(directory_path):
    list_file = os.listdir(directory_path)
    list_pdf = []
    for file in list_file:
        if file.endswith("pdf"):
            list_pdf.append(file)
    df = pd.DataFrame(columns=["title", "paragraphs"])
    for i, pdf in enumerate(list_pdf):
        try:
            df.loc[i] = [pdf, None]
            raw = parser.from_file(os.path.join(directory_path, pdf))
            s = raw["content"]
            paragraphs = re.split(u"\n(?=\u2028|[A-Z-0-9])", s)
            list_par = []
            for p in paragraphs:
                if len(p) >= 200:
                    list_par.append(p.replace("\n", ""))
                df.loc[i, "paragraphs"] = list_par
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Unable to process file {}".format(pdf))
    return df


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return "".join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def md_converter(directory_path):
    """Get all md, convert them to html and create the pandas dataframe with columns ['title', 'paragraphs']"""
    dict_doc = {"title": [], "paragraphs": []}
    for md_file in Path(directory_path).glob("**/*.md"):
        md_file = str(md_file)
        filename = md_file.split("/")[-1]
        try:
            with open(md_file, "r") as f:
                dict_doc["title"].append(filename)
                md_text = f.read()
                html_text = markdown.markdown(md_text)
                html_text_list = list(html_text.split("<p>"))
                for i in range(len(html_text_list)):
                    html_text_list[i] = (
                        strip_tags(html_text_list[i])
                        .replace("\n", " ")
                        .lstrip()
                        .rstrip()
                    )
                clean_text_list = list(filter(None, html_text_list))
                dict_doc["paragraphs"].append(clean_text_list)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Unable to process file {}".format(filename))
    df = pd.DataFrame.from_dict(dict_doc)
    return df
