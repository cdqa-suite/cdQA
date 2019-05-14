# cdQA <img src="https://cdn.discordapp.com/emojis/513893717816508416.gif" width="40" height="40"/>

[![Build Status](https://travis-ci.com/fmikaelian/cdQA.svg?token=Vzy9RRKRZ41ynd9q2BRX&branch=develop)](https://travis-ci.com/fmikaelian/cdQA)
[![codecov](https://codecov.io/gh/fmikaelian/cdQA/branch/develop/graph/badge.svg?token=F16X0IU6RT)](https://codecov.io/gh/fmikaelian/cdQA)
[![PyPI Version](https://img.shields.io/pypi/v/cdqa.svg)](https://pypi.org/project/cdqa/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/cdqa.svg)](https://pypi.org/project/cdqa/)
[![Binder](https://mybinder.org/badge.svg)]()
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

An end-to-end closed-domain question answering system with BERT and classic IR methods 📚

- [Installation](#installation)
  - [With pip](#with-pip)
  - [From source](#from-source)
  - [Hardware Requirements](#hardware-requirements)
- [Getting started](#getting-started)
  - [How it works](#how-it-works)
  - [Preparing your data](#preparing-your-data)
  - [Training models](#training-models)
  - [Making predictions](#making-predictions)
  - [Evaluating models](#evaluating-models)
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

### How it works

<img src="docs/how-it-works.png" width="30%"/>

### Preparing your data

To use `cdqa` you need a `.csv` corpus file with the following columns:

| date     | title             | category             | link                         | abstract             | paragraphs                                       | content                                       |
| -------- | ----------------- | -------------------- | ---------------------------- | -------------------- | ------------------------------------------------ | --------------------------------------------- |
| DD/MM/YY | The Article Title | The Article Category | https://the-article-link.com | The Article Abstract | [Paragraph 1 of Article, Paragraph N of Article] | Paragraph 1 of Article Paragraph N of Article |

You can use the converters to create this file:

```python
# create a corpus dataframe from a directory containing .pdf files
```

### Training models

Read your corpus in `.csv` format:

```python
import pandas as pd
from cdqa.pipeline.cdqa_sklearn import QAPipeline

df = pd.read_csv('your-custom-corpus-here.csv')
```

Fit the pipeline on your corpus using the pre-trained reader:

```python
cdqa_pipeline = QAPipeline(model='bert_qa_squad_v1.1_sklearn.joblib')
cdqa_pipeline.fit(X=df)
```

If you want to fine-tune the reader on your custom data:

```python
cdqa_pipeline = QAPipeline()
cdqa_pipeline.fit(X=df, fit_reader=True)
```

### Making predictions

To get the best prediction given an input query:

```python
query = 'your custom question here'

cdqa_pipeline.predict(X=query)
```

### Evaluating models

In order to evaluate models on your custom dataset you will need to annotate it. The annotation process can be done in 3 steps:

1. Convert your pandas DataFrame into a json file with SQuAD format:

    ```python
    from cdqa.utils.converter import df2squad

    json_data = df2squad(df=df, squad_version='v2.0', output_dir='../data', filename='bnpp_newsroom-v1.1')
    ```

2. Use an annotator to add ground truth question-answer pairs:

    Please refer to [`cdQA-annotator`](https://github.com/fmikaelian/cdQA-annotator), a web-based annotator for closed-domain question answering datasets with SQuAD format.

3. Evaluate your model:

    ```python
    from cdqa.utils.metrics import evaluate, evaluate_from_files

    evaluate(dataset, predictions) # as json objects

    evaluate_from_files(dataset_file='dev-v1.1.json', prediction_file='predictions.json') # as json files
    ```

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
http localhost:5000/api query=='your question here'
```

If you wish to serve a user interface, follow the instructions of [cdQA-ui](https://github.com/fmikaelian/cdQA-ui), a web interface developed for `cdQA`.

### With docker

You can use the [Dockerfile](Dockerfile) to deploy the full `cdQA` app.

## Contributing

Read our [Contributing Guidelines](CONTRIBUTING.md).

## References

| Type                 | Title                                                                                                                | Author                                                                                 | Year |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---- |
| :video_camera: Video | [Stanford CS224N: NLP with Deep Learning Lecture 10 – Question Answering](https://youtube.com/watch?v=yIdF-17HwSk)   | Christopher Manning                                                                    | 2019 |
| :newspaper: Paper    | [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718)                        | Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, Jimmy Lin | 2019 |
| :newspaper: Paper    | [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006)                       | Noah A. Smith                                                                          | 2019 |
| :newspaper: Paper    | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                           | 2018 |
| :newspaper: Paper    | [Neural Reading Comprehension and Beyond](https://cs.stanford.edu/people/danqi/papers/thesis.pdf)                    | Danqi Chen                                                                             | 2018 |
| :newspaper: Paper    | [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)                                | Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes                                   | 2017 |