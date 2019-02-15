import pandas as pd
import prettytable
import time
from sklearn.feature_extraction.text import TfidfVectorizer

def train_document_retriever(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def predict_document_retriever(question, vectorizer, tfidf_matrix, top_n, verbose=True):
    question_vector = vectorizer.transform([question])
    scores = pd.DataFrame(tfidf_matrix.dot(question_vector.T).toarray())
    closest_docs_indices = scores.sort_values(by=0, ascending=False).index[:top_n].to_list()
    if verbose:
        process(closest_docs_indices)
    return(closest_docs_indices)

def process(closest_docs_indices):
    t0 = time.time()
    rank = 1
    table = prettytable.PrettyTable(['rank', 'index', 'title'])
    for i in range(len(closest_docs_indices)):
        index = closest_docs_indices[i]
        title = df.iloc[int(index)]['title']
        table.add_row([rank, index, title])
        rank+=1
    print(table)
    print('Time: {} seconds'.format(time.time() - t0))