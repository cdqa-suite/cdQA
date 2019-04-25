# cdQA

[![Build Status](https://travis-ci.com/fmikaelian/cdQA.svg?token=Vzy9RRKRZ41ynd9q2BRX&branch=develop)](https://travis-ci.com/fmikaelian/cdQA) [![codecov](https://codecov.io/gh/fmikaelian/cdQA/branch/develop/graph/badge.svg?token=F16X0IU6RT)](https://codecov.io/gh/fmikaelian/cdQA)
 [![License](
https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

An end-to-end closed-domain question answering system with BERT and classic IR methods 📚

- [Installation](#installation)
  - [With pip](#with-pip)
  - [From source](#from-source)
  - [Hardware Requirements](#hardware-requirements)
- [Getting started](#getting-started)
  - [Preparing your data](#preparing-your-data)
  - [Training models](#training-models)
  - [Using models](#using-models)
  - [Evaluating models](#evaluating-models)
  - [Downloading pre-trained models](#downloading-pre-trained-models)
  - [Practical examples](#practical-examples)
- [Deployment](#deployment)
  - [Manual](#manual)
  - [With docker](#with-docker)
- [Contributing](#contributing)
- [References](#references)

## Installation

### With pip

```shell
pip install cdqa
```

### From source

```shell
git clone https://github.com/fmikaelian/cdQA.git
cd cdQA
pip install -e .
```

### Hardware Requirements

Experiments have been done on an AWS EC2 `p3.2xlarge` Deep Learning AMI (Ubuntu) Version 22.0 + a single Tesla V100 16GB with 16-bits training enabled (to accelerate training and prediction). To enable this feature, you will need to install [`apex`](https://github.com/nvidia/apex):

```shell
git clone https://github.com/NVIDIA/apex.git
cd apex/
python setup.py install --cuda_ext --cpp_ext
```

## Getting started

### Preparing your data

To use `cdqa` on a custom corpus you need to convert this corpus into a `pandas.DataFrame` with the following columns:

| date     | title             | category             | link                         | abstract             | paragraphs                                           | content                                         |
| -------- | ----------------- | -------------------- | ---------------------------- | -------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| DD/MM/YY | The Article Title | The Article Category | https://the-article-link.com | The Article Abstract | [Paragraph 1 of Article, Paragraph N of Article] | Paragraph 1 of Article Paragraph N of Article |

### Training models

First we train the document retriever:

```python
import pandas as pd
from cdqa.retriever.tfidf_doc_ranker import train_document_retriever

df = pd.read_csv('your-custom-corpus.csv')
article_vectorizer, article_tfidf_matrix = train_document_retriever(corpus=df['content'])
```

Then the document reader:

```python
from cdqa.reader.bertqa_sklearn import BertProcessor, BertQA

train_processor = BertProcessor(bert_model='bert-base-uncased', do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X='data/train-v1.1.json')

model = BertQA(bert_model='bert-base-uncased',
               train_batch_size=12,
               learning_rate=3e-5,
               num_train_epochs=2,
               do_lower_case=True,
               fp16=True,
               output_dir='models/bert_qa_squad_v1.1_sklearn')

model.fit(X=(train_examples, train_features))
```

### Using models

First the document retriever finds the most relevant documents given an input question:

```python
from cdqa.retriever.tfidf_doc_ranker import predict_document_retriever
from cdqa.utils.converter import generate_squad_examples

question = 'Ask your question here'

article_indices = predict_document_retriever(question=question,
                                             paragraphs=None,
                                             vectorizer=article_vectorizer,
                                             tfidf_matrix=article_tfidf_matrix,
                                             top_n=3,
                                             metadata=df,
                                             verbose=True)
```

Then these documents are processed:

```python
squad_examples = generate_squad_examples(question=question,
                                         article_indices=article_indices,
                                         metadata=df)

test_processor = BertProcessor(bert_model='bert-base-uncased', do_lower_case=True, is_training=False)
test_examples, test_features = test_processor.fit_transform(X=squad_examples)
```

Finally the document reader finds the best answer among the retrieved documents:

```python
final_prediction, all_predictions, all_nbest_json, scores_diff_json = model.predict(X=(test_examples, test_features))

print(question, final_prediction)
```

### Evaluating models

In order to evaluate models on your custom dataset you will need to annotate it. For this you can use [`cdQA-annotator`](https://github.com/fmikaelian/cdQA-annotator), a web-based annotator for closed-domain question answering datasets with SQuAD format. Then you can split your dataset into train and test sets as follows and run the evaluation script.

```python
python evaluate-v1.1.py data/cdqa-v1.1.json logs/your_model/predictions.json
```

### Downloading pre-trained models

To download existing data and models automatically from the Github releases, you will need a personal Github token. You can find [how to create one here](https://github.com/settings/tokens) (you only need to select the `repo` scope). Save your token as an environment variable:

```shell
export token='YOUR_GITHUB_TOKEN'
```

You can now execute the `download.py` to get all Github release assets:

```shell
python cdqa/pipeline/download.py
```

The data is saved in  `/data` and the models in `/models`. You can load the models with `joblib.load`.


### Practical examples

A complete worfklow is described in our [`examples`](examples) notebook.

## Deployment

### Manual

You can deploy a `cdQA` REST API by executing:

```shell
FLASK_APP=api.py flask run -h 0.0.0.0
```

To try it, execute:

```shell
http localhost:5000/api q=='your question here'
```

If you wish to serve a user interface, follow the instructions of [cdQA-ui](https://github.com/fmikaelian/cdQA-ui), a web interface developed for `cdQA`.

### With docker

You can use the [Dockerfile](Dockerfile) to deploy the full `cdQA` app.

## Contributing

Read our [Contributing Guidelines](CONTRIBUTING.md).

## References

| Type              | Title                                                                                                                | Author                                                                                 | Year |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---- |
| :newspaper: Paper | [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718)                        | Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, Jimmy Lin | 2019 |
| :newspaper: Paper | [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006)                       | Noah A. Smith                                                                          | 2019 |
| :newspaper: Paper | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                           | 2018 |
| :newspaper: Paper | [Neural Reading Comprehension and Beyond](https://cs.stanford.edu/people/danqi/papers/thesis.pdf)                    | Danqi Chen                                                                             | 2018 |
| :newspaper: Paper | [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)                                | Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes                                   | 2017 |
