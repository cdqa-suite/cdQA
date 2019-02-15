import pandas as pd
from reading-comprehension.retriever.tfidf_doc_ranker import train_document_retriever, predict_document_retriever

df = pd.read_csv('data.csv')

vectorizer, tfidf_matrix = train_document_retriever(corpus=df['content'])

document_retriever_indexes = predict_document_retriever(question='machine learning',
                                                        vectorizer=vectorizer,
                                                        tfidf_matrix=tfidf_matrix,
                                                        top_n=3,
                                                        verbose=True)