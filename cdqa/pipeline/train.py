import pandas as pd
from ast import literal_eval
from joblib import dump

from cdqa.utils.converter import filter_paragraphs
from cdqa.retriever.tfidf_doc_ranker import train_document_retriever
from cdqa.reader.bertqa_sklearn import BertProcessor, BertQA

# train document retriever
df = pd.read_csv('data/bnpp_newsroom_v1.0/bnpp_newsroom-v1.0.csv', converters={'paragraphs': literal_eval})
df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)
df['content'] = df['paragraphs'].apply(lambda x: ' '.join(x))
article_vectorizer, article_tfidf_matrix = train_document_retriever(corpus=df['content'])
dump(article_vectorizer, 'models/article_vectorizer.joblib')
dump(article_tfidf_matrix, 'models/article_tfidf_matrix.joblib')

# train document reader
train_processor = BertProcessor(bert_model='bert-base-uncased', is_training=True)
train_examples, train_features = train_processor.fit_transform(X='data/train-v1.1.json')

model = BertQA(bert_model='bert-base-uncased',
               custom_weights=False,
               train_batch_size=12,
               learning_rate=3e-5,
               num_train_epochs=2,
               output_dir='logs/bert_qa_squad_v1.1_sklearn')

model.fit(X=(train_examples, train_features))

dump(model, 'model.joblib')