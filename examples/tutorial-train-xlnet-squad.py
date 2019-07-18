import wget
import os
import torch
from sklearn.externals import joblib
from cdqa.reader.reader_sklearn import Reader

# download SQuAD 2.0 assets
squad_urls = [
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
]

for squad_url in squad_urls:
    wget.download(url=squad_url, out='.')

# cast Reader class with train params
reader = Reader(model_type='xlnet',
                model_name_or_path='xlnet-base-cased',
                fp16=False,
                output_dir='.')

# train the model
reader.fit(X='train-v2.0.json')

# save GPU version locally
joblib.dump(reader, os.path.join(reader.output_dir, 'xlnet_qa_vGPU.joblib'))

# send current reader model to CPU
reader.model.to('cpu')
reader.device = torch.device('cpu')

# save CPU it locally
joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_vCPU.joblib'))

# evaluate the model
reader.evaluate(X='dev-v2.0.json')