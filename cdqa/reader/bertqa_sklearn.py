# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import uuid
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from pytorch_pretrained_bert.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME,
    CONFIG_NAME,
)
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import (
    BasicTokenizer,
    BertTokenizer,
    whitespace_tokenize,
)

from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
        paragraph=None,
        title=None,
        retriever_score=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.paragraph = paragraph
        self.title = title
        self.retriever_score = retriever_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, version_2_with_negative, n_jobs=-1):
    """Read a SQuAD json file into a list of SquadExample."""

    if isinstance(input_file, str):
        with open(input_file, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]
    else:
        input_data = input_file

    # Read examples with multiprocessing over entries
    processes = n_jobs if n_jobs != -1 else mp.cpu_count()
    with mp.Pool(processes=processes) as pool:
        examples = pool.map(
            _read_entry_parallel,
            [(entry, is_training, version_2_with_negative) for entry in input_data],
        )

    # examples will be a nested list, unflattenning with no parallelization
    # showed to be more effective

    return _flatten_examples(examples)


def _flatten_examples(examples):
    flatten_examples = []
    for entry in examples:
        for processed_qa in entry:
            flatten_examples.append(processed_qa)
    return flatten_examples


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _read_entry_parallel(args_tuple):

    entry, is_training, version_2_with_negative = args_tuple

    examples = []
    for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            try:
                retriever_score = qa["retriever_score"]
            except KeyError:
                retriever_score = 0
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if is_training:
                if version_2_with_negative:
                    is_impossible = qa["is_impossible"]
                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                        "For training, each question should have exactly 1 answer."
                    )
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[
                        answer_offset + answer_length - 1
                    ]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position : (end_position + 1)]
                    )
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text)
                    )
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning(
                            "Could not find answer: '%s' vs. '%s'",
                            actual_text,
                            cleaned_answer_text,
                        )
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            examples.append(
                SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    paragraph=paragraph_text,
                    title=entry["title"],
                    retriever_score=retriever_score,
                )
            )
    return examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    verbose,
    n_jobs=-1,
):
    """Loads a data file into a list of `InputBatch`s."""

    # Read examples with multiprocessing over examples
    processes = n_jobs if n_jobs != -1 else mp.cpu_count()
    with mp.Pool(processes=processes) as pool:
        features = pool.map(
            _example_to_features_parallel,
            [
                (
                    example_index,
                    example,
                    tokenizer,
                    max_seq_length,
                    doc_stride,
                    max_query_length,
                    is_training,
                    verbose,
                )
                for (example_index, example) in enumerate(examples)
            ],
        )

    # features will be a nested list, unflattenning with no parallelization
    # showed to be more effective

    return _flatten_features(features)


def _flatten_features(features):
    flatten_features = []
    for feat_list in features:
        for feat in feat_list:
            flatten_features.append(feat)
    return flatten_features


def _example_to_features_parallel(args_tuple):
    (
        example_index,
        example,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        verbose,
    ) = args_tuple

    features = []

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            example.orig_answer_text,
        )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"]
    )
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        unique_id = uuid.uuid4().int
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(
                doc_spans, doc_span_index, split_token_index
            )
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0
        if example_index < 20 and verbose:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("doc_span_index: %s" % (doc_span_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info(
                "token_to_orig_map: %s"
                % " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()])
            )
            logger.info(
                "token_is_max_context: %s"
                % " ".join(
                    ["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]
                )
            )
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if is_training and example.is_impossible:
                logger.info("impossible example")
            if is_training and not example.is_impossible:
                answer_text = " ".join(tokens[start_position : (end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info("answer: %s" % (answer_text))

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
            )
        )

    return features


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def write_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    retriever_score_weight,
    n_predictions=None,
):
    """
    Write final predictions to the json file and log-odds of null if needed.
    It returns:
        - if n_predictions == None: a tuple (best_prediction, final_predictions)
        - if n_predictions != None: a tuple (best_prediction, final_predictions, n_best_predictions_list)
    """
    if verbose_logging:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    final_predictions = []

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = {}
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
                all_nbest_json[example.qas_id] = nbest_json

        best_dict = nbest_json[0]
        best_dict["qas_id"] = example.qas_id
        best_dict["title"] = example.title
        best_dict["paragraph"] = example.paragraph
        best_dict["retriever_score"] = example.retriever_score.item()
        best_dict["final_score"] = (1 - retriever_score_weight) * (
            best_dict["start_logit"] + best_dict["end_logit"]
        ) + retriever_score_weight * best_dict["retriever_score"]
        final_predictions.append(best_dict)

    final_predictions_sorted = sorted(
        final_predictions, key=lambda d: d["final_score"], reverse=True
    )

    best_prediction = (
        final_predictions_sorted[0]["text"],
        final_predictions_sorted[0]["title"],
        final_predictions_sorted[0]["paragraph"],
        final_predictions_sorted[0]["final_score"],
    )

    return_list = [best_prediction, final_predictions_sorted]

    if n_predictions:
        n_best_predictions_list = _n_best_predictions(
            final_predictions_sorted, n_predictions
        )
        return_list.append(n_best_predictions_list)

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    if version_2_with_negative and output_null_log_odds_file:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return tuple(return_list)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text,
                tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class BertProcessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer to convert SQuAD examples to BertQA input format.

    Parameters
    ----------
    bert_version : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
        bert-base-multilingual-cased, bert-base-chinese.
    do_lower_case : bool, optional
        Whether to lower case the input text. True for uncased models, False for cased models.
        Default: True
    is_training : bool, optional
        Whether you are in training phase.
    version_2_with_negative : bool, optional
        If true, the SQuAD examples contain some that do not have an answer.
    max_seq_length : int, optional
        The maximum total input sequence length after WordPiece tokenization. Sequences
        longer than this will be truncated, and sequences shorter than this will be padded.
    doc_stride : int, optional
        When splitting up a long document into chunks, how much stride to take between chunks.
    max_query_length : int, optional
        The maximum number of tokens for the question. Questions longer than this will
        be truncated to this length.
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.
    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. 'None' means 1 unless in a
        :obj:'joblib.parallel_backend' context. '-1' means using all processors.

    Returns
    -------
    examples : list
        SquadExample
    features : list
        InputFeatures

    Examples
    --------
    >>> from cdqa.reader import BertProcessor
    >>> processor = BertProcessor(bert_model='bert-base-uncased', do_lower_case=True, is_training=False)
    >>> examples, features = processor.fit_transform(X=squad_examples)

    """

    def __init__(
        self,
        bert_model="bert-base-uncased",
        do_lower_case=True,
        is_training=False,
        version_2_with_negative=False,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        verbose=False,
        tokenizer=None,
        n_jobs=1,
    ):

        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.is_training = is_training
        self.version_2_with_negative = version_2_with_negative
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.verbose = verbose
        self.n_jobs = n_jobs

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.bert_model, do_lower_case=self.do_lower_case
            )
        else:
            self.tokenizer = tokenizer
            logger.info("loading custom tokenizer")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        examples = read_squad_examples(
            input_file=X,
            is_training=self.is_training,
            version_2_with_negative=self.version_2_with_negative,
            n_jobs=self.n_jobs,
        )

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=self.is_training,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

        return examples, features


class BertQA(BaseEstimator):
    """
    A scikit-learn estimator for BertForQuestionAnswering.

    Parameters
    ----------
    bert_model : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
        bert-base-multilingual-cased, bert-base-chinese.
    train_batch_size : int, optional
        Total batch size for training. (the default is 32)
    predict_batch_size : int, optional
        Total batch size for predictions. (the default is 8)
    learning_rate : float, optional
        The initial learning rate for Adam. (the default is 5e-5)
    num_train_epochs : float, optional
        Total number of training epochs to perform. (the default is 3.0)
    warmup_proportion : float, optional
        Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%%
        of training. (the default is 0.1)
    n_best_size : int, optional
        The total number of n-best predictions to generate in the nbest_predictions.json
        output file. (the default is 20)
    max_answer_length : int, optional
        The maximum length of an answer that can be generated. This is needed because the start
        and end predictions are not conditioned on one another. (the default is 30)
    verbose_logging : bool, optional
        If true, all of the warnings related to data processing will be printed.
        A number of warnings are expected for a normal SQuAD evaluation. (the default is False)
    no_cuda : bool, optional
        Whether not to use CUDA when available (the default is False)
    seed : int, optional
        random seed for initialization (the default is 42)
    gradient_accumulation_steps : int, optional
        Number of updates steps to accumulate before performing a backward/update pass. (the default is 1)
    do_lower_case : bool, optional
        Whether to lower case the input text. True for uncased models, False for cased models. (the default is True)
    local_rank : int, optional
        local_rank for distributed training on gpus (the default is -1)
    fp16 : bool, optional
        Whether to use 16-bit float precision instead of 32-bit (the default is False)
    loss_scale : int, optional
        Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
        0 (default value): dynamic loss scaling.
        Positive power of 2: static loss scaling value. (the default is 0)
    version_2_with_negative : bool, optional
        If true, the SQuAD examples contain some that do not have an answer. (the default is False)
    null_score_diff_threshold : float, optional
        If null_score - best_non_null is greater than the threshold predict null. (the default is 0.0)
    output_dir : str, optional
        The output directory where the model checkpoints and predictions will be written.
        If None, nothing is saved. (the default is None)
    server_ip : str, optional
        Can be used for distant debugging. (the default is '')
    server_port : str, optional
        Can be used for distant debugging. (the default is '')


    Attributes
    ----------
    device : torch.device
        [description]
    n_gpu : int
        [description]
    model : pytorch_pretrained_bert.modeling.BertForQuestionAnswering
        [description]

    Examples
    --------
    >>> from cdqa.reader import BertQA
    >>> model = BertQA(bert_model='bert-base-uncased',
                train_batch_size=12,
                learning_rate=3e-5,
                num_train_epochs=2,
                do_lower_case=True,
                fp16=True,
                output_dir='models/bert_qa_squad_v1.1_sklearn')
    >>> model.fit(X=(train_examples, train_features))
    >>> final_prediction = model.predict(X=(test_examples, test_features))

    """

    def __init__(
        self,
        bert_model="bert-base-uncased",
        train_batch_size=32,
        predict_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3.0,
        warmup_proportion=0.1,
        n_best_size=20,
        max_answer_length=30,
        verbose_logging=False,
        no_cuda=False,
        seed=42,
        gradient_accumulation_steps=1,
        do_lower_case=True,
        local_rank=-1,
        fp16=False,
        loss_scale=0,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        output_dir=None,
        server_ip="",
        server_port="",
    ):

        self.bert_model = bert_model
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.verbose_logging = verbose_logging
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.do_lower_case = do_lower_case
        self.local_rank = local_rank
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        self.output_dir = output_dir
        self.server_ip = server_ip
        self.server_port = server_port

        # Prepare model
        self.model = BertForQuestionAnswering.from_pretrained(
            self.bert_model,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE),
                "distributed_{}".format(self.local_rank),
            ),
        )

        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd

            print("Waiting for debugger attach")
            ptvsd.enable_attach(
                address=(self.server_ip, self.server_port), redirect_output=True
            )
            ptvsd.wait_for_attach()

        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
            )
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")

        if self.verbose_logging:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN,
            )

            logger.info(
                "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                    self.device, self.n_gpu, bool(self.local_rank != -1), self.fp16
                )
            )

    def fit(self, X, y=None):

        train_examples, train_features = X

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    self.gradient_accumulation_steps
                )
            )

        self.train_batch_size = (
            self.train_batch_size // self.gradient_accumulation_steps
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if self.fp16:
            self.model.half()
        self.model.to(self.device)
        if self.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            self.model = DDP(self.model)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0

        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        )
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long
        )
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_start_positions,
            all_end_positions,
        )
        if self.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.train_batch_size
        )

        num_train_optimization_steps = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )

        if self.local_rank != -1:
            num_train_optimization_steps = (
                num_train_optimization_steps // torch.distributed.get_world_size()
            )

        if self.verbose_logging:
            logger.info("***** Running training *****")
            logger.info("  Num orig examples = %d", len(train_examples))
            logger.info("  Num split examples = %d", len(train_features))
            logger.info("  Batch size = %d", self.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                bias_correction=False,
                max_grad_norm=1.0,
            )
            if self.loss_scale == 0:
                optimizer = FP16_Optimizer(
                    optimizer, dynamic_loss_scale=True, verbose=False
                )
            else:
                optimizer = FP16_Optimizer(
                    optimizer, static_loss_scale=self.loss_scale, verbose=False
                )
            warmup_linear = WarmupLinearSchedule(
                warmup=self.warmup_proportion, t_total=num_train_optimization_steps
            )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                warmup=self.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

        self.model.train()
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc="Iteration",
                    disable=self.local_rank not in [-1, 0],
                )
            ):
                if self.n_gpu == 1:
                    batch = tuple(
                        t.to(self.device) for t in batch
                    )  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = (
                    batch
                )
                loss = self.model(
                    input_ids, segment_ids, input_mask, start_positions, end_positions
                )
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if self.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if self.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = self.learning_rate * warmup_linear.get_lr(
                            global_step, self.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and configuration
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        if self.output_dir:
            output_model_file = os.path.join(self.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(self.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

        self.model.to(self.device)

        return self

    def predict(
        self, X, n_predictions=None, retriever_score_weight=0.35, return_all_preds=False
    ):

        eval_examples, eval_features = X
        if self.verbose_logging:
            logger.info("***** Running predictions *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", self.predict_batch_size)

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index
        )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.predict_batch_size
        )

        self.model.to(self.device)
        self.model.eval()
        all_results = []
        if self.verbose_logging:
            logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
            if len(all_results) % 1000 == 0 and self.verbose_logging:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = self.model(
                    input_ids, segment_ids, input_mask
                )
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits,
                    )
                )
        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            output_prediction_file = os.path.join(self.output_dir, "predictions.json")
            output_nbest_file = os.path.join(self.output_dir, "nbest_predictions.json")
            output_null_log_odds_file = os.path.join(self.output_dir, "null_odds.json")
        else:
            output_prediction_file = None
            output_nbest_file = None
            output_null_log_odds_file = None

        result_tuple = write_predictions(
            eval_examples,
            eval_features,
            all_results,
            self.n_best_size,
            self.max_answer_length,
            self.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            self.verbose_logging,
            self.version_2_with_negative,
            self.null_score_diff_threshold,
            retriever_score_weight,
            n_predictions,
        )

        if n_predictions is not None:
            return result_tuple[-1]

        best_prediction, final_predictions = result_tuple

        if return_all_preds:
            return final_predictions

        return best_prediction


def _n_best_predictions(final_predictions_sorted, n):
    n = min(n, len(final_predictions_sorted))
    final_prediction_list = []
    for i in range(n):
        curr_pred = (
            final_predictions_sorted[i]["text"],
            final_predictions_sorted[i]["title"],
            final_predictions_sorted[i]["paragraph"],
            final_predictions_sorted[i]["final_score"],
        )
        final_prediction_list.append(curr_pred)
    return final_prediction_list
