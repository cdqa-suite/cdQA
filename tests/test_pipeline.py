import pytest
from ast import literal_eval
import pandas as pd
import torch

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import *
from cdqa.pipeline import QAPipeline

def execute_pipeline(model, query, n_predictions=None):
    download_bnpp_data("./data/bnpp_newsroom_v1.1/")
    download_model(model + "-squad_1.1", dir="./models")
    df = pd.read_csv(
        "./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv",
        converters={"paragraphs": literal_eval},
    )
    df = filter_paragraphs(df)

    reader_path = "models/" + model + "_qa.joblib"

    cdqa_pipeline = QAPipeline(reader=reader_path)
    cdqa_pipeline.fit_retriever(df)

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

@pytest.mark.parametrize("model", ["bert", "distilbert"])
def test_predict(model):
    assert execute_pipeline(model,
        "Since when does the Excellence Program of BNP Paribas exist?"
    ) == ("January 2016", "BNP Paribas’ commitment to universities and schools")


def test_n_predictions():
    predictions = execute_pipeline("distilbert",
        "Since when does the Excellence Program of BNP Paribas exist?", 5
    )

    assert len(predictions) == 5

    assert predictions[0] == (
        "January 2016",
        "BNP Paribas’ commitment to universities and schools",
    )
