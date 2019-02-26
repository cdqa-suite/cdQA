from reader.run_squad import BertProcessor, BertQA
from joblib import dump

train_processor = BertProcessor(is_training=True)
train_examples, train_features = train_processor.fit_transform(X='samples/dev-v1.1.json')

model = BertQA(bert_model='models/bert_qa_squad_v1.1')
model.fit(X_y=train)

dump(model, 'model.joblib')