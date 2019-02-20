import pandas as pd
from cdqa.retriever.tfidf_doc_ranker import train_document_retriever, predict_document_retriever

# https://stackoverflow.com/questions/32742976/how-to-read-a-column-of-csv-as-dtype-list-using-pandas
df = pd.read_csv('bnpp_newsroom_v1.0.csv', converters={'paragraphs': literal_eval})

vectorizer, tfidf_matrix = train_document_retriever(corpus=df['paragraphs'])

document_retriever_indexes = predict_document_retriever(question='machine learning',
                                                        vectorizer=vectorizer,
                                                        tfidf_matrix=tfidf_matrix,
                                                        top_n=3,
                                                        verbose=True)