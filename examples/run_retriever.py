import pandas as pd
from cdqa.utils.converter import filter_paragraphs
from cdqa.retriever.tfidf_doc_ranker import train_document_retriever, predict_document_retriever
from ast import literal_eval
from tqdm import tqdm

# https://stackoverflow.com/questions/32742976/how-to-read-a-column-of-csv-as-dtype-list-using-pandas
df = pd.read_csv('data/bnpp_newsroom_v1.0/bnpp_newsroom-v1.0.csv', converters={'paragraphs': literal_eval})

df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)

# document retrivial with article granularity
df['content'] = df['paragraphs'].apply(lambda x: ' '.join(x))

article_vectorizer, article_tfidf_matrix = train_document_retriever(corpus=df['content'])
article_indices = predict_document_retriever(question='artificial intelligence',
                                             paragraphs=None,
                                             vectorizer=article_vectorizer,
                                             tfidf_matrix=article_tfidf_matrix,
                                             top_n=3,
                                             metadata=df,
                                             verbose=True)

# document retrivial with paragraph granularity
paragraphs = []
for index, row in tqdm(df.iterrows()):
    for paragraph in row['paragraphs']:
        paragraphs.append({'index': index, 'context': paragraph})

paragraph_vectorizer, paragraph_tfidf_matrix = train_document_retriever(corpus=[paragraph['context'] for paragraph in paragraphs])
paragraphs_indices = predict_document_retriever(question='artificial intelligence',
                                                paragraphs=paragraphs,
                                                vectorizer=paragraph_vectorizer,
                                                tfidf_matrix=paragraph_tfidf_matrix,
                                                top_n=3,
                                                metadata=df,
                                                verbose=True)
