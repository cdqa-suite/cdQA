from ast import literal_eval
import pandas as pd
import torch

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import *
from cdqa.pipeline import QAPipeline


def execute_pipeline(query, n_predictions=None):
    download_bnpp_data("./data/bnpp_newsroom_v1.1/")
    download_model("bert-squad_1.1", dir="./models")
    df = pd.read_csv(
        "./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv",
        converters={"paragraphs": literal_eval},
    )
    df = filter_paragraphs(df)

    cdqa_pipeline = QAPipeline(reader="models/bert_qa_vCPU-sklearn.joblib")
    cdqa_pipeline.fit_retriever(df)
    if torch.cuda.is_available():
        cdqa_pipeline.cuda()
    if n_predictions is not None:
        predictions = cdqa_pipeline.predict(query, n_predictions=n_predictions)
        result = []

        for answer, title, paragraph, score in predictions:
            prediction = (answer, title)
            result.append(prediction)
        return result
    else:
        prediction = cdqa_pipeline.predict(query)
        result = (prediction[0], prediction[1])
        return result


def test_predict():
    assert execute_pipeline(
        "Since when does the Excellence Program of BNP Paribas exist?"
    ) == ("January 2016", "BNP Paribas’ commitment to universities and schools")


def test_n_predictions():
    predictions = execute_pipeline(
        "Since when does the Excellence Program of BNP Paribas exist?", 5
    )

    assert len(predictions) == 5

    assert predictions[0] == (
        "January 2016",
        "BNP Paribas’ commitment to universities and schools",
    )
