from reader.bertqa_sklearn import BertProcessor, BertQA
from joblib import load

test_processor = BertProcessor(is_training=False)
test_examples, test_features = test_processor.fit_transform(X='samples/dev-v1.1.json')

print("  Num orig examples = %d", len(test_examples))
print("  Num split examples = %d", len(test_features))

model = load('model.joblib') 

model.predict(X=test_features)