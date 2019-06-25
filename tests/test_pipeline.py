import os
from ast import literal_eval
import pandas as pd

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline.cdqa_sklearn import QAPipeline


def execute_pipeline(query):
    df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv',
                     converters={'paragraphs': literal_eval})
    df = filter_paragraphs(df)

    cdqa_pipeline = QAPipeline(
        reader='models/bert_qa_vCPU-sklearn.joblib')
    cdqa_pipeline.fit(X=df)
    cdqa_pipeline.reader.output_dir = None

    prediction = cdqa_pipeline.predict(X=query)

    result = (prediction[0], prediction[1])

    return result


def test_predict():
    assert execute_pipeline('Since when does the Excellence Program of BNP Paribas exist?') == (
        'January 2016', 'BNP Paribas’ commitment to universities and schools')
