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
    ngram_range : bool, optional
        [description] (the default is False)
    max_df : bool, optional
        [description] (the default is False)
    stop_words : bool, optional
        [description] (the default is False)
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
    top_n : int
        maximum number of top articles to retrieve
        header should be of format: date, title, category, link, abstract, paragraphs, content.
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
                 ngram_range=(1, 2),
                 max_df=0.85,
                 stop_words='english',
                 paragraphs=None,
                 top_n=3,
                 verbose=True):

        self.ngram_range = ngram_range
        self.max_df = max_df
        self.stop_words = stop_words
        self.paragraphs = paragraphs
        self.top_n = top_n
        self.verbose = verbose

    def fit(self, X, y=None):

        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                          max_df=self.max_df,
                                          stop_words=self.stop_words)
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
