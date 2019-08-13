import pandas as pd
import prettytable
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

class TfidfRetriever(BaseEstimator):
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
    top_n : int
        maximum number of top articles to retrieve
        header should be of format: title, paragraphs.
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
    >>> from cdqa.retriever.tfidf_retriever_sklearn import TfidfRetriever

    >>> retriever = TfidfRetriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(X=df['content'])
    >>> closest_docs_indices = retriever.predict(X='Since when does the the Excellence Program of BNP Paribas exist?')

    >>> paragraphs = []
    >>> for index, row in tqdm(df.iterrows()):
    >>>     for paragraph in row['paragraphs']:
    >>>         paragraphs.append({'index': index, 'context': paragraph})

    >>> retriever = TfidfRetriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(X=[paragraph['context'] for paragraph in paragraphs])
    >>> closest_docs_indices = retriever.predict(X='Since when does the the Excellence Program of BNP Paribas exist?')

    """

    def __init__(self,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words='english',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 2),
                 max_df=0.85,
                 min_df=2,
                 vocabulary=None,
                 paragraphs=None,
                 top_n=3,
                 verbose=False):

        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary
        self.paragraphs = paragraphs
        self.top_n = top_n
        self.verbose = verbose

    def fit(self, X, y=None):

        self.vectorizer = TfidfVectorizer(lowercase=self.lowercase,
                                          preprocessor=self.preprocessor,
                                          tokenizer=self.tokenizer,
                                          stop_words=self.stop_words,
                                          token_pattern=self.token_pattern,
                                          ngram_range=self.ngram_range,
                                          max_df=self.max_df,
                                          min_df=self.min_df,
                                          vocabulary=self.vocabulary
                                          )
        self.tfidf_matrix = self.vectorizer.fit_transform(X)

        return self

    def predict(self, X, metadata):

        t0 = time.time()
        question_vector = self.vectorizer.transform([X])
        scores = pd.DataFrame(self.tfidf_matrix.dot(question_vector.T).toarray())
        closest_docs_indices = scores.sort_values(by=0, ascending=False).index[:self.top_n].values

        # inspired from https://github.com/facebookresearch/DrQA/blob/50d0e49bb77fe0c6e881efb4b6fe2e61d3f92509/scripts/reader/interactive.py#L63
        if self.verbose:
            rank = 1
            table = prettytable.PrettyTable(['rank', 'index', 'title'])
            for i in range(len(closest_docs_indices)):
                index = closest_docs_indices[i]
                if self.paragraphs:
                    article_index = self.paragraphs[int(index)]['index']
                    title = metadata.iloc[int(article_index)]['title']
                else:
                    title = metadata.iloc[int(index)]['title']
                table.add_row([rank, index, title])
                rank+=1
            print(table)
            print('Time: {} seconds'.format(round(time.time() - t0, 5)))

        return closest_docs_indices
