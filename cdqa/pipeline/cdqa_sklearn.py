import joblib

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

from cdqa.retriever.tfidf_sklearn import TfidfRetriever
from cdqa.utils.converter import filter_paragraphs, generate_squad_examples
from cdqa.reader.bertqa_sklearn import BertProcessor, BertQA


class QAPipeline(BaseEstimator):
    """
    A scikit-learn implementation of the whole cdQA pipeline

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe containing your corpus of documents metadata
        header should be of format: date, title, category, link, abstract, paragraphs, content.
    model : str or .joblib object of a version of BERT model with sklearn wrapper, optional
    bert_version : str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
        bert-base-multilingual-cased, bert-base-chinese.


    Examples
    --------
    >>> from cdqa.pipeline.qa_pipeline import QAPipeline
    >>> qa_pipe = QAPipeline(model='bert_qa_squad_vCPU-sklearn.joblib', metadata=df)
    >>> qa_pipe.fit()
    >>> prediction = qa_pipe.predict(X='When BNP Paribas was created?')

    >>> from cdqa.pipeline.qa_pipeline import QAPipeline
    >>> qa_pipe = QAPipeline(metadata=df)
    >>> qa_pipe.fit('train-v1.1.json', fit_reader=True)
    >>> qa_pipe.fit()
    >>> prediction = qa_pipe.predict(X='When BNP Paribas was created?')

    """

    def __init__(self, metadata, model=None, bert_version='bert-base-uncased', **kwargs):

        # Separating kwargs
        kwargs_bertqa = {key: value for key, value in kwargs.items()
                         if key in BertQA.__init__.__code__.co_varnames}

        kwargs_processor = {key: value for key, value in kwargs.items()
                            if key in BertProcessor.__init__.__code__.co_varnames}

        kwargs_retriever = {key: value for key, value in kwargs.items()
                            if key in TfidfRetriever.__init__.__code__.co_varnames}

        if not model:
            self.model = BertQA(self.bert_version, **kwargs_bertqa)
        elif type(model) == str:
            self.model = joblib.load(model)
        else:
            self.model = model

        self.metadata = metadata
        self.bert_version = bert_version

        self.processor_train = BertProcessor(self.bert_version,
                                             is_training=True,
                                             **kwargs_processor)

        self.processor_predict = BertProcessor(self.bert_version,
                                               is_training=False,
                                               **kwargs_processor)

        self.retriever = TfidfRetriever(self.metadata, **kwargs_retriever)

    def fit(self, X=None, y=None, fit_reader=False):
        """ Fit the QAPipeline retriever to a list of documents in a dataframe if fit_reader is false,
        fit the reader (QABert model) to a json file squad-like with questions and answers

        Parameters
        ----------
        X: dict or str
            Dictionaire with questions and answers in SQUAD format or path to json file in SQUAD format
        fit_reader: boolean, default false
            Whether to fit reader (BertQA model) or retriever

        """
        if not fit_reader:
            self.retriever.fit(self.metadata['content'])
        else:
            if not X:
                raise RuntimeError(
                    'fit_reader is True, please pass a json file in SQUAD format as input')
            train_examples, train_features = self.processor_train.fit_transform(X)
            self.model.fit(X=(train_examples, train_features))

        return self

    def predict(self, X):
        """ Compute prediction of an answer to a question

        """

        closest_docs_indices = self.retriever.predict(X)
        squad_examples = generate_squad_examples(question=X,
                                                 closest_docs_indices=closest_docs_indices,
                                                 metadata=self.metadata)
        examples, features = self.processor_predict.fit_transform(X=squad_examples)
        prediction = self.model.predict((examples, features))

        return prediction
