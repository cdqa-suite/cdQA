""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import torch
import string
import re
import argparse
import tqdm
import json
import sys
import os

import joblib
from tqdm.autonotebook import tqdm


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, unique_pred=True):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                if unique_pred:
                    prediction = predictions[qa["id"]]
                    increm_em = metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths
                    )
                    increm_f1 = metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths
                    )
                else:
                    preds = predictions[qa["id"]]
                    increm_em = max(
                        [
                            metric_max_over_ground_truths(
                                exact_match_score, prediction, ground_truths
                            )
                            for prediction in preds
                        ]
                    )
                    increm_f1 = max(
                        [
                            metric_max_over_ground_truths(
                                f1_score, prediction, ground_truths
                            )
                            for prediction in preds
                        ]
                    )

                exact_match += increm_em
                f1 += increm_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def evaluate_reader(cdqa_pipeline, dataset_file, expected_version="1.1"):
    """Evaluation for SQuAD

    Parameters
    ----------
    cdqa_pipeline: QAPipeline object
        Pipeline with reader to be evaluated
    dataset_file : str
        path to json file in SQuAD format
    expected_version : str, optional
        [description], by default '1.1'

    Returns
    -------
    A dictionary with exact match and f1 scores
    """

    with open(dataset_file, "r") as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json["version"] != expected_version:
            print(
                "Evaluation expects v-"
                + expected_version
                + ", but got dataset with v-"
                + dataset_json["version"],
                file=sys.stderr,
            )
        dataset = dataset_json["data"]

    if torch.cuda.is_available():
        cdqa_pipeline.cuda()
    reader = cdqa_pipeline.reader
    processor = cdqa_pipeline.processor_predict
    examples, features = processor.fit_transform(dataset)
    preds = reader.predict((examples, features), return_all_preds=True)
    all_predictions = {d['qas_id']: d['text'] for d in preds}

    return evaluate(dataset, all_predictions)


def evaluate_pipeline(
    cdqa_pipeline,
    annotated_json,
    output_dir="./results",
    n_predictions=None,
    verbose=True,
):
    """Evaluation method for a whole pipeline (retriever + reader)

    Parameters
    ----------
    cdqa_pipeline: QAPipeline object
        Pipeline to be evaluated
    annotated_json: str
        path to json file in SQuAD format with annotated questions and answers
    output_dir: str
        path to directory where results and predictions will be saved. If None,
        no file will be saved
    verbose: boolean
        whether the result should be printed or not

    Returns
    -------
    A dictionary with exact match and f1 scores

    """
    if output_dir is not None:
        dir = os.path.expanduser(output_dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = os.path.join(dir, annotated_json.split("/")[-1][:-5])
        if not os.path.exists(dir):
            os.makedirs(dir)
        preds_path = os.path.join(dir, "all_predictions.json")
        results_path = os.path.join(dir, "results.json")

    with open(annotated_json, "r") as file:
        data_dict = json.load(file)

    queries = _get_queries_list(data_dict)
    all_predictions = _pipeline_predictions(cdqa_pipeline, queries, n_predictions)
    if output_dir is not None:
        with open(preds_path, "w") as f:
            json.dump(all_predictions, f)

    unique_pred = n_predictions is None
    results = evaluate(data_dict["data"], all_predictions, unique_pred)
    if output_dir is not None:
        with open(results_path, "w") as f:
            json.dump(results, f)
    if verbose:
        print("\nEvaluation results:", results)

    return results


def _get_queries_list(data_dict):

    queries = []
    articles = data_dict["data"]
    for article in articles:
        paragraphs = article["paragraphs"]
        for paragraph in paragraphs:
            questions = paragraph["qas"]
            for question in questions:
                query = question["question"]
                id = question["id"]
                queries.append((id, query))

    return queries


def _pipeline_predictions(cdqa_pipeline, queries, n_predictions=None):

    all_predictions = dict()
    for id, query in tqdm(queries):
        if n_predictions is None:
            all_predictions[id] = cdqa_pipeline.predict(query)[0]
        else:
            preds = cdqa_pipeline.predict(query, n_predictions=n_predictions)
            all_predictions[id] = [pred[0] for pred in preds]
    return all_predictions
