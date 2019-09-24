import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .text_transformers import BM25Transformer


class BM25Vectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of BM25 features and computes
    scores of the documents based on a query

    Vectorizer inspired on the sklearn.feature_extraction.text.TfidfVectorizer
    class

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'} (default='strict')
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default=None)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

        .. versionchanged:: 0.21
        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
        first read from the file and then passed to the given callable
        analyzer.

    stop_words : string {'english'}, list, or None (default=None)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=None)
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean (default=False)
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional (default=float64)
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`

    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.

    k1 : float, optional (default=2.0)
        term k1 in the BM25 formula

    b : float, optional (default=0.75)
        term b in the BM25 formula

    floor : float or None, optional (default=None)
        floor value for idf terms

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    idf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.
    """

    def __init__(
        self,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm=None,
        use_idf=True,
        k1=2.0,
        b=0.75,
        floor=None,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._bm25 = BM25Transformer(norm, use_idf, k1, b)

    # Broadcast the BM25 parameters to the underlying transformer instance
    # for easy grid search and repr
    @property
    def norm(self):
        return self._bm25.norm

    @norm.setter
    def norm(self, value):
        self._bm25.norm = value

    @property
    def use_idf(self):
        return self._bm25.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._bm25.use_idf = value

    @property
    def k1(self):
        return self._bm25.k1

    @k1.setter
    def k1(self, value):
        self._bm25.k1 = value

    @property
    def b(self):
        return self._bm25.b

    @b.setter
    def b(self, value):
        self._bm25.b = value

    @property
    def idf_(self):
        return self._bm25.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal "
                    "to vocabulary size = %d" % (len(value), len(self.vocabulary))
                )
        self._bm25.idf_ = value

    def fit(self, raw_documents, y=None):
        """
        Learn vocabulary and BM25 stats from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : BM25Vectorizer
        """
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self

    def transform(self, raw_corpus, is_query=False):
        """
        Vectorizes the input, whether it is a query or the list of documents

        Parameters
        ----------
        raw_corpus : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors : sparse matrix, [n_queries, n_documents]
            scores from BM25 statics for each document with respect to each query
        """
        X = super().transform(raw_corpus) if is_query else None

        return self._bm25.transform(X, copy=False, is_query=is_query)

    def fit_transform(self, raw_documents, y=None):
        """
        Learn vocabulary, idf and BM25 features. Return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
            
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            BM25 document-term matrix.
        """
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self._bm25.transform(X, copy=False)
