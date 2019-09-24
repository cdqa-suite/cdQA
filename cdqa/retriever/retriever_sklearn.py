import pandas as pd
import prettytable
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from .vectorizers import BM25Vectorizer


class BaseRetriever(BaseEstimator, ABC):
    """
    Abstract base class for all Retriever classes.
    All retrievers should inherit from this class.
    Each retriever class should implement a _fit_vectorizer method and a
    _compute_scores method
    """

    def __init__(self, vectorizer, top_n=10, verbose=False):
        self.vectorizer = vectorizer
        self.top_n = top_n
        self.verbose = verbose

    def fit(self, df: pd.DataFrame, y=None):
        """
        Fit the retriever to a list of documents or paragraphs

        Parameters
        ----------
        df: pandas.DataFrame object with all documents
        """
        self.metadata = df
        return self._fit_vectorizer(df)

    @abstractmethod
    def _fit_vectorizer(self, df):
        pass

    @abstractmethod
    def _compute_scores(self, query):
        pass

    def predict(self, query: str) -> OrderedDict:
        """
        Compute the top_n closest documents given a query

        Parameters
        ----------
        query: str

        Returns
        -------
        best_idx_scores: OrderedDict
            Dictionnaire with top_n best scores and idices of the documents as keys

        """
        t0 = time.time()
        scores = self._compute_scores(query)
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        best_idx_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)[: self.top_n]
        )

        # inspired from https://github.com/facebookresearch/DrQA/blob/50d0e49bb77fe0c6e881efb4b6fe2e61d3f92509/scripts/reader/interactive.py#L63
        if self.verbose:
            rank = 1
            table = prettytable.PrettyTable(["rank", "index", "title"])
            for i in range(len(closest_docs_indices)):
                index = closest_docs_indices[i]
                if self.paragraphs:
                    article_index = self.paragraphs[int(index)]["index"]
                    title = self.metadata.iloc[int(article_index)]["title"]
                else:
                    title = self.metadata.iloc[int(index)]["title"]
                table.add_row([rank, index, title])
                rank += 1
            print(table)
            print("Time: {} seconds".format(round(time.time() - t0, 5)))

        return best_idx_scores


class TfidfRetriever(BaseRetriever):
    """
    A scikit-learn estimator for TfidfRetriever. Trains a tf-idf matrix from a corpus
    of documents then finds the most N similar documents of a given input document by
    taking the dot product of the vectorized input document and the trained tf-idf matrix.

    Parameters
    ----------
    lowercase : boolean
        Convert all characters to lowercase before tokenizing. (default is True)
    preprocessor : callable or None
        Override the preprocessing (string transformation) stage while preserving
        the tokenizing and n-grams generation steps. (default is None)
    tokenizer : callable or None
        Override the string tokenization step while preserving the preprocessing
        and n-grams generation steps (default is None)
    stop_words : string {‘english’}, list, or None
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. ‘english’ is currently the only supported string value.
        If a list, that list is assumed to contain stop words, all of which will
        be removed from the resulting tokens.
        If None, no stop words will be used. max_df can be set to a value in the
        range [0.7, 1.0) to automatically detect and filter stop words based on
        intra corpus document frequency of terms.
        (default is None)
    token_pattern : string
        Regular expression denoting what constitutes a “token”. The default regexp
        selects tokens of 2 or more alphanumeric characters (punctuation is completely
        ignored and always treated as a token separator).
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different n-grams
        to be extracted. All values of n such that min_n <= n <= max_n will be used.
        (default is (1, 1))
    max_df : float in range [0.0, 1.0] or int
        When building the vocabulary ignore terms that have a document frequency strictly
        higher than the given threshold (corpus-specific stop words). If float, the parameter
        represents a proportion of documents, integer absolute counts. This parameter is
        ignored if vocabulary is not None. (default is 1.0)
    min_df : float in range [0.0, 1.0] or int
        When building the vocabulary ignore terms that have a document frequency
        strictly lower than the given threshold. This value is also called cut-off
        in the literature. If float, the parameter represents a proportion of
        documents, integer absolute counts. This parameter is ignored if vocabulary
        is not None. (default is 1)
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices
        in the feature matrix, or an iterable over terms. If not given, a vocabulary
        is determined from the input documents. (default is None)
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
    top_n : int (default 20)
        maximum number of top articles (or paragraphs) to retrieve
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.

    Examples
    --------
    >>> from cdqa.retriever import TfidfRetriever

    >>> retriever = TfidfRetriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(X=df)
    >>> best_idx_scores = retriever.predict(X='Since when does the the Excellence Program of BNP Paribas exist?')
    """

    def __init__(
        self,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        vocabulary=None,
        top_n=20,
        verbose=False,
    ):
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary

        vectorizer = TfidfVectorizer(
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            tokenizer=self.tokenizer,
            stop_words=self.stop_words,
            token_pattern=self.token_pattern,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            vocabulary=self.vocabulary,
        )
        super().__init__(vectorizer, top_n, verbose)

    def _fit_vectorizer(self, df, y=None):
        self.tfidf_matrix = self.vectorizer.fit_transform(df["content"])
        return self

    def _compute_scores(self, query):
        question_vector = self.vectorizer.transform([query])
        scores = self.tfidf_matrix.dot(question_vector.T).toarray()
        return scores


class BM25Retriever(BaseRetriever):
    """
    A scikit-learn estimator for BM25Retriever. Trains a matrix based on BM25 statistics
    from a corpus of documents then finds the most N similar documents of a given input
    query by computing the BM25 score for each document based on the query.

    Parameters
    ----------
    lowercase : boolean
        Convert all characters to lowercase before tokenizing. (default is True)
    preprocessor : callable or None
        Override the preprocessing (string transformation) stage while preserving
        the tokenizing and n-grams generation steps. (default is None)
    tokenizer : callable or None
        Override the string tokenization step while preserving the preprocessing
        and n-grams generation steps (default is None)
    stop_words : string {‘english’}, list, or None
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. ‘english’ is currently the only supported string value.
        If a list, that list is assumed to contain stop words, all of which will
        be removed from the resulting tokens.
        If None, no stop words will be used. max_df can be set to a value in the
        range [0.7, 1.0) to automatically detect and filter stop words based on
        intra corpus document frequency of terms.
        (default is None)
    token_pattern : string
        Regular expression denoting what constitutes a “token”. The default regexp
        selects tokens of 2 or more alphanumeric characters (punctuation is completely
        ignored and always treated as a token separator).
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different n-grams
        to be extracted. All values of n such that min_n <= n <= max_n will be used.
        (default is (1, 1))
    max_df : float in range [0.0, 1.0] or int
        When building the vocabulary ignore terms that have a document frequency strictly
        higher than the given threshold (corpus-specific stop words). If float, the parameter
        represents a proportion of documents, integer absolute counts. This parameter is
        ignored if vocabulary is not None. (default is 1.0)
    min_df : float in range [0.0, 1.0] or int
        When building the vocabulary ignore terms that have a document frequency
        strictly lower than the given threshold. This value is also called cut-off
        in the literature. If float, the parameter represents a proportion of
        documents, integer absolute counts. This parameter is ignored if vocabulary
        is not None. (default is 1)
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices
        in the feature matrix, or an iterable over terms. If not given, a vocabulary
        is determined from the input documents. (default is None)
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
    top_n : int (default 20)
        maximum number of top articles (or paragraphs) to retrieve
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.
    k1 : float, optional (default=2.0)
        term k1 in the BM25 formula
    b : float, optional (default=0.75)
        term b in the BM25 formula
    floor : float or None, optional (default=None)
        floor value for idf terms

    Attributes
    ----------
    vectorizer : BM25Vectorizer

    Examples
    --------
    >>> from cdqa.retriever import BM25Retriever

    >>> retriever = BM25Retriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(df=df)
    >>> best_idx_scores = retriever.predict(query='Since when does the the Excellence Program of BNP Paribas exist?')

    """

    def __init__(
        self,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        vocabulary=None,
        top_n=20,
        verbose=False,
        k1=2.0,
        b=0.75,
        floor=None,
    ):

        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b
        self.floor = floor

        vectorizer = BM25Vectorizer(
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            tokenizer=self.tokenizer,
            stop_words=self.stop_words,
            token_pattern=self.token_pattern,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            vocabulary=self.vocabulary,
            k1=self.k1,
            b=self.b,
            floor=self.floor,
        )
        super().__init__(vectorizer, top_n, verbose)

    def _fit_vectorizer(self, df, y=None):
        self.bm25_matrix = self.vectorizer.fit_transform(df["content"])
        return self

    def _compute_scores(self, query):
        question_vector = self.vectorizer.transform([query], is_query=True)
        scores = self.bm25_matrix.dot(question_vector.T).toarray()
        return scores
