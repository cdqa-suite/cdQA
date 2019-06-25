import os
import torch
from sklearn.externals import joblib
from cdqa.reader.bertqa_sklearn import BertProcessor, BertQA

# pre-process examples
train_processor = BertProcessor(do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X='data/train-v1.1.json')

# train the model
reader = BertQA(train_batch_size=12,
               learning_rate=3e-5,
               num_train_epochs=2,
               do_lower_case=True,
               fp16=False,
               output_dir='models')

reader.fit(X=(train_examples, train_features))

# save GPU version locally
joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_vGPU.joblib'))

# send current reader model to CPU
reader.model.to('cpu')
reader.device = torch.device('cpu')

# save CPU it locally
joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_vCPU.joblib'))