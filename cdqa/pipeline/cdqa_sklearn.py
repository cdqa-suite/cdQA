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
    metadata: pandas.DataFrame
        dataframe containing your corpus of documents metadata
        header should be of format: date, title, category, link, abstract, paragraphs, content.
    reader: str (path to .joblib) or .joblib object of an instance of BertQA (BERT model with sklearn wrapper), optional
    bert_version: str
        Bert pre-trained model selected in the list: bert-base-uncased,
        bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
        bert-base-multilingual-cased, bert-base-chinese.
    kwargs: kwargs for BertQA(), BertProcessor() and TfidfRetriever()
        Please check documentation for these classes


    Examples
    --------
    >>> from cdqa.pipeline.qa_pipeline import QAPipeline
    >>> qa_pipeline = QAPipeline(reader='bert_qa_squad_vCPU-sklearn.joblib')
    >>> qa_pipeline.fit(X=df)
    >>> prediction = qa_pipeline.predict(X='When BNP Paribas was created?')

    >>> from cdqa.pipeline.qa_pipeline import QAPipeline
    >>> qa_pipeline = QAPipeline()
    >>> qa_pipeline.fit_reader('train-v1.1.json')
    >>> qa_pipeline.fit(X=df)
    >>> prediction = qa_pipeline.predict(X='When BNP Paribas was created?')

    """

    def __init__(self, reader=None, **kwargs):

        # Separating kwargs
        kwargs_bertqa = {key: value for key, value in kwargs.items()
                         if key in BertQA.__init__.__code__.co_varnames}

        kwargs_processor = {key: value for key, value in kwargs.items()
                            if key in BertProcessor.__init__.__code__.co_varnames}

        kwargs_retriever = {key: value for key, value in kwargs.items()
                            if key in TfidfRetriever.__init__.__code__.co_varnames}

        if not reader:
            self.reader = BertQA(**kwargs_bertqa)
        elif type(reader) == str:
            self.reader = joblib.load(reader)
        else:
            self.reader = reader

        self.processor_train = BertProcessor(is_training=True,
                                             **kwargs_processor)

        self.processor_predict = BertProcessor(is_training=False,
                                               **kwargs_processor)

        self.retriever = TfidfRetriever(**kwargs_retriever)

    def fit(self, X=None, y=None):
        """ Fit the QAPipeline retriever to a list of documents in a dataframe.

        Parameters
        ----------
        X: pandas.Dataframe
            Dataframe with the following columns: "title", "paragraphs" and "content"

        """
        
        self.metadata = X
        self.retriever.fit(self.metadata['content'])        

        return self

    def fit_reader(self, X=None, y=None):
        """Train the reader (BertQA instance) of QAPipeline object

        Parameters
        ----------
        X = path to json file in SQUAD format

        """

        train_examples, train_features = self.processor_train.fit_transform(X)
        self.reader.fit(X=(train_examples, train_features))

        return self

    def predict(self, X=None):
        """ Compute prediction of an answer to a question

        Parameters
        ----------
        X = str
            Sample (question) to perform a prediction on

        """

        closest_docs_indices = self.retriever.predict(X, metadata=self.metadata)
        squad_examples = generate_squad_examples(question=X,
                                                 closest_docs_indices=closest_docs_indices,
                                                 metadata=self.metadata)
        examples, features = self.processor_predict.fit_transform(X=squad_examples)
        prediction = self.reader.predict((examples, features))

        return prediction
