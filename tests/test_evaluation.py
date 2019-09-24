import json
from ast import literal_eval
import pandas as pd
import torch

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.evaluation import evaluate_pipeline
from cdqa.utils.download import *
from cdqa.pipeline import QAPipeline


def test_evaluate_pipeline():

    download_bnpp_data("./data/bnpp_newsroom_v1.1/")
    download_model("bert-squad_1.1", dir="./models")
    df = pd.read_csv(
        "./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv",
        converters={"paragraphs": literal_eval},
    )
    df = filter_paragraphs(df)

    test_data = {
        "data": [
            {
                "title": "BNP Paribas’ commitment to universities and schools",
                "paragraphs": [
                    {
                        "context": "Since January 2016, BNP Paribas has offered an Excellence Program targeting new Master’s level graduates (BAC+5) who show high potential. The aid program lasts 18 months and comprises three assignments of six months each. It serves as a strong career accelerator that enables participants to access high-level management positions at a faster rate. The program allows participants to discover the BNP Paribas Group and its various entities in France and abroad, build an internal and external network by working on different assignments and receive personalized assistance from a mentor and coaching firm at every step along the way.",
                        "qas": [
                            {
                                "answers": [
                                    {"answer_start": 6, "text": "January 2016"},
                                    {"answer_start": 6, "text": "January 2016"},
                                    {"answer_start": 6, "text": "January 2016"},
                                ],
                                "question": "Since when does the Excellence Program of BNP Paribas exist?",
                                "id": "56be4db0acb8001400a502ec",
                            }
                        ],
                    }
                ],
            }
        ],
        "version": "1.1",
    }

    with open("./test_data.json", "w") as f:
        json.dump(test_data, f)

    cdqa_pipeline = QAPipeline(reader="./models/bert_qa_vCPU-sklearn.joblib", n_jobs=-1)
    cdqa_pipeline.fit_retriever(df)
    if torch.cuda.is_available():
        cdqa_pipeline.cuda()

    eval_dict = evaluate_pipeline(cdqa_pipeline, "./test_data.json", output_dir=None)

    assert eval_dict["exact_match"] > 0.8

    assert eval_dict["f1"] > 0.8
