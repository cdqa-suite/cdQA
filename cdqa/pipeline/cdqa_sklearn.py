import joblib
import warnings

import pandas as pd
import numpy as np
import torch

from sklearn.base import BaseEstimator

from cdqa.retriever import TfidfRetriever, BM25Retriever
from cdqa.utils.converters import generate_squad_examples
from cdqa.reader import BertProcessor, BertQA

RETRIEVERS = {"bm25": BM25Retriever, "tfidf": TfidfRetriever}


class QAPipeline(BaseEstimator):
    """
    A scikit-learn implementation of the whole cdQA pipeline

    Parameters
    ----------
    reader: str (path to .joblib) or .joblib object of an instance of BertQA (BERT model with sklearn wrapper), optional

    retriever: "bm25" or "tfidf"
        The type of retriever

    retrieve_by_doc: bool (default: True). If Retriever will rank by documents
        or by paragraphs.

    kwargs: kwargs for BertQA(), BertProcessor(), TfidfRetriever() and BM25Retriever()
        Please check documentation for these classes

    Examples
    --------
    >>> from cdqa.pipeline import QAPipeline
    >>> qa_pipeline = QAPipeline(reader='bert_qa_squad_vCPU-sklearn.joblib')
    >>> qa_pipeline.fit_retriever(X=df)
    >>> prediction = qa_pipeline.predict(X='When BNP Paribas was created?')

    >>> from cdqa.pipeline import QAPipeline
    >>> qa_pipeline = QAPipeline()
    >>> qa_pipeline.fit_reader('train-v1.1.json')
    >>> qa_pipeline.fit_retriever(X=df)
    >>> prediction = qa_pipeline.predict(X='When BNP Paribas was created?')

    """

    def __init__(self, reader=None, retriever="bm25", retrieve_by_doc=False, **kwargs):

        if retriever not in RETRIEVERS:
            raise ValueError(
                "You provided a type of retriever that is not supported. "
                + "Please provide a retriver in the following list: "
                + str(list(RETRIEVERS.keys()))
            )

        retriever_class = RETRIEVERS[retriever]

        # Separating kwargs
        kwargs_bertqa = {
            key: value
            for key, value in kwargs.items()
            if key in BertQA.__init__.__code__.co_varnames
        }

        kwargs_processor = {
            key: value
            for key, value in kwargs.items()
            if key in BertProcessor.__init__.__code__.co_varnames
        }

        kwargs_retriever = {
            key: value
            for key, value in kwargs.items()
            if key in retriever_class.__init__.__code__.co_varnames
        }

        if not reader:
            self.reader = BertQA(**kwargs_bertqa)
        elif type(reader) == str:
            self.reader = joblib.load(reader)
        else:
            self.reader = reader

        self.processor_train = BertProcessor(is_training=True, **kwargs_processor)

        self.processor_predict = BertProcessor(is_training=False, **kwargs_processor)

        self.retriever = retriever_class(**kwargs_retriever)

        self.retrieve_by_doc = retrieve_by_doc

        if torch.cuda.is_available():
            self.cuda()

    def fit_retriever(self, df: pd.DataFrame = None):
        """ Fit the QAPipeline retriever to a list of documents in a dataframe.
        Parameters
        ----------
        df: pandas.Dataframe
            Dataframe with the following columns: "title", "paragraphs"
        """

        if self.retrieve_by_doc:
            self.metadata = df
            self.metadata["content"] = self.metadata["paragraphs"].apply(
                lambda x: " ".join(x)
            )
        else:
            self.metadata = self._expand_paragraphs(df)

        self.retriever.fit(self.metadata)

        return self

    def fit_reader(self, data=None):
        """ Fit the QAPipeline retriever to a list of documents in a dataframe.

        Parameters
        ----------
        data: dict str-path to json file
             Annotated dataset in squad-like for Reader training

        """

        train_examples, train_features = self.processor_train.fit_transform(data)
        self.reader.fit(X=(train_examples, train_features))

        return self

    def predict(
        self,
        query: str = None,
        n_predictions: int = None,
        retriever_score_weight: float = 0.35,
        return_all_preds: bool = False,
    ):
        """ Compute prediction of an answer to a question

        Parameters
        ----------
        X: str
            Sample (question) to perform a prediction on

        n_predictions: int or None (default: None).
            Number of returned predictions. If None, only one prediction is return

        retriever_score_weight: float (default: 0.35).
            The weight of retriever score in the final score used for prediction.
            Given retriever score and reader average of start and end logits, the final score used for ranking is:

            final_score = retriever_score_weight * retriever_score + (1 - retriever_score_weight) * (reader_avg_logit)

        return_all_preds: boolean (default: False)
            whether to return a list of all predictions done by the Reader or not

        Returns
        -------
        if return_all_preds is False:
        prediction: tuple (answer, title, paragraph, score/logit)

        if return_all_preds is True:
        List of dictionnaries with all metadada of all answers outputted by the Reader
        given the question.

        """

        if not isinstance(query, str):
            raise TypeError(
                "The input is not a string. Please provide a string as input."
            )
        if not (
            isinstance(n_predictions, int) or n_predictions is None or n_predictions < 1
        ):
            raise TypeError("n_predictions should be a positive Integer or None")
        best_idx_scores = self.retriever.predict(query)
        squad_examples = generate_squad_examples(
            question=query,
            best_idx_scores=best_idx_scores,
            metadata=self.metadata,
            retrieve_by_doc=self.retrieve_by_doc,
        )
        examples, features = self.processor_predict.fit_transform(X=squad_examples)
        prediction = self.reader.predict(
            X=(examples, features),
            n_predictions=n_predictions,
            retriever_score_weight=retriever_score_weight,
            return_all_preds=return_all_preds,
        )
        return prediction

    def to(self, device):
        """ Send reader to CPU if device=='cpu' or to GPU if device=='cuda'
        """
        if device not in ("cpu", "cuda"):
            raise ValueError("Attribute device should be 'cpu' or 'cuda'.")

        self.reader.model.to(device)
        self.reader.device = torch.device(device)
        return self

    def cpu(self):
        """ Send reader to CPU
        """
        self.reader.model.cpu()
        self.reader.device = torch.device("cpu")
        return self

    def cuda(self):
        """ Send reader to GPU
        """
        self.reader.model.cuda()
        self.reader.device = torch.device("cuda")
        return self

    def dump_reader(self, filename):
        """ Dump reader model to a .joblib object
        """
        self.cpu()
        joblib.dump(self.reader, filename)
        if torch.cuda.is_available():
            self.cuda()

    @staticmethod
    def _expand_paragraphs(df):
        # Snippet taken from: https://stackoverflow.com/a/48532692/11514226
        lst_col = "paragraphs"
        df = pd.DataFrame(
            {
                col: np.repeat(df[col].values, df[lst_col].str.len())
                for col in df.columns.drop(lst_col)
            }
        ).assign(**{lst_col: np.concatenate(df[lst_col].values)})[df.columns]
        df["content"] = df["paragraphs"]
        return df.drop("paragraphs", axis=1)
