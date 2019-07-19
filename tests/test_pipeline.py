import os
from ast import literal_eval
import wget
import pandas as pd

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline.cdqa_sklearn import QAPipeline


def load_data():
    df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv',
                     converters={'paragraphs': literal_eval})
    df = filter_paragraphs(df)
    return df


def fit_retriever():
    df = load_data()
    cdqa_pipeline = QAPipeline(
        reader='models/bert_qa_vCPU-sklearn.joblib')
    cdqa_pipeline.fit(X=df)
    return cdqa_pipeline


def fit_reader():

    cdqa_pipeline = fit_retriever()
    wget.download(url='https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/examples/tests_samples/SQUAD/dev-v2.0-small.json',
                  out='data')
    cdqa_pipeline.fit_reader(X='data/dev-v2.0-small.json')
    return cdqa_pipeline


def make_prediction(query):

    cdqa_pipeline = fit_retriever()
    cdqa_pipeline.reader.output_dir = None
    prediction = cdqa_pipeline.predict(X=query)

    result = (prediction[0], prediction[1])

    return result


def make_prediction_with_reader_finetuning(query):

    cdqa_pipeline = fit_reader()
    cdqa_pipeline.reader.output_dir = None
    prediction = cdqa_pipeline.predict(X=query)

    result = (prediction[0], prediction[1])

    return result


def test_predict():
    assert make_prediction('Since when does the Excellence Program of BNP Paribas exist?') == (
        'January 2016', 'BNP Paribas’ commitment to universities and schools')


def test_predict_with_reader_finetuning():
    assert make_prediction_with_reader_finetuning('Since when does the Excellence Program of BNP Paribas exist?') == (
        'January 2016', 'BNP Paribas’ commitment to universities and schools')
