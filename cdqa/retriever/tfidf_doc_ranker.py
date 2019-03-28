import pandas as pd
import prettytable
import time
from sklearn.feature_extraction.text import TfidfVectorizer

def train_document_retriever(corpus):
    """
    trains a tf-idf matrix from a corpus of documents
    
    Parameters
    ----------
    corpus : iterable
        an iterable which yields either str, unicode or file objects
    
    Returns
    -------
    vectorizer : TfidfVectorizer
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.

    Examples
    --------
    >>> 

    """

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def predict_document_retriever(question, paragraphs, vectorizer, tfidf_matrix, top_n, metadata, verbose=True):
    """
    finds the most N similar documents of a given input document by taking
    the dot product of the vectorized input document and the trained tf-idf matrix

    Parameters
    ----------
    question : str
        input question or query
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
        
    vectorizer : TfidfTransformer
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.
    top_n : int
        maximum number of top articles to retrieve
    metadata : pandas.DataFrame
        dataframe containing your corpus of documents metadata
        header should be of format: date, title, category, link, abstract, paragraphs, content.
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.

    Examples
    --------
    >>> 

    """


    t0 = time.time()
    question_vector = vectorizer.transform([question])
    scores = pd.DataFrame(tfidf_matrix.dot(question_vector.T).toarray())
    closest_docs_indices = scores.sort_values(by=0, ascending=False).index[:top_n].values
    
    # inspired from https://github.com/facebookresearch/DrQA/blob/50d0e49bb77fe0c6e881efb4b6fe2e61d3f92509/scripts/reader/interactive.py#L63
    if verbose:
        rank = 1
        table = prettytable.PrettyTable(['rank', 'index', 'title'])
        for i in range(len(closest_docs_indices)):
            index = closest_docs_indices[i]
            if paragraphs:
                article_index = paragraphs[int(index)]['index']
                title = metadata.iloc[int(article_index)]['title']
            else:
                title = metadata.iloc[int(index)]['title']
            table.add_row([rank, index, title])
            rank+=1
        print(table)
        print('Time: {} seconds'.format(round(time.time() - t0, 5)))

    return(closest_docs_indices)
