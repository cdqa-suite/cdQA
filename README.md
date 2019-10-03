# cdQA: Closed Domain Question Answering

[![Build Status](https://travis-ci.com/cdqa-suite/cdQA.svg?branch=master)](https://travis-ci.com/cdqa-suite/cdQA)
[![codecov](https://codecov.io/gh/cdqa-suite/cdQA/branch/master/graph/badge.svg)](https://codecov.io/gh/cdqa-suite/cdQA)
[![PyPI Version](https://img.shields.io/pypi/v/cdqa.svg)](https://pypi.org/project/cdqa/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/cdqa.svg)](https://pypi.org/project/cdqa/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cdqa-suite/cdQA/master?filepath=examples%2Ftutorial-first-steps-cdqa.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdqa-suite/cdQA/blob/master/examples/tutorial-first-steps-cdqa.ipynb)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](.github/CODE_OF_CONDUCT.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![GitHub](https://img.shields.io/github/license/cdqa-suite/cdQA.svg)

An End-To-End Closed Domain Question Answering System. Built on top of the HuggingFace [transformers](https://github.com/huggingface/transformers) library.

## cdQA in details

If you are interested in understanding how the system works and its implementation, we wrote an [article on Medium](https://towardsdatascience.com/how-to-create-your-own-question-answering-system-easily-with-python-2ef8abc8eb5) with a high-level explanation.

We also made a presentation during the \#9 NLP Breakfast organised by [Feedly](feedly.com). You can check it out [here](https://blog.feedly.com/nlp-breakfast-9-closed-domain-question-answering/).

## Table of Contents <!-- omit in toc -->

- [Installation](#Installation)
  - [With pip](#With-pip)
  - [From source](#From-source)
  - [Hardware Requirements](#Hardware-Requirements)
- [Getting started](#Getting-started)
  - [Preparing your data](#Preparing-your-data)
    - [Manual](#Manual)
    - [With converters](#With-converters)
  - [Downloading pre-trained models](#Downloading-pre-trained-models)
  - [Training models](#Training-models)
  - [Making predictions](#Making-predictions)
  - [Evaluating models](#Evaluating-models)
- [Notebook Examples](#Notebook-Examples)
- [Deployment](#Deployment)
  - [Manual](#Manual-1)
- [Contributing](#Contributing)
- [References](#References)
- [LICENSE](#LICENSE)

## Installation

### With pip

```shell
pip install cdqa
```

### From source

```shell
git clone https://github.com/cdqa-suite/cdQA.git
cd cdQA
pip install -e .
```

### Hardware Requirements

Experiments have been done with:

- **CPU** 👉 AWS EC2 `t2.medium` Deep Learning AMI (Ubuntu) Version 22.0
- **GPU** 👉 AWS EC2 `p3.2xlarge` Deep Learning AMI (Ubuntu) Version 22.0 + a single Tesla V100 16GB.

## Getting started

### Preparing your data

#### Manual

To use `cdQA` you need to create a pandas dataframe with the following columns:

| title             | paragraphs                                             |
| ----------------- | ------------------------------------------------------ |
| The Article Title | [Paragraph 1 of Article, ... , Paragraph N of Article] |

#### With converters

The objective of `cdqa` converters is to make it easy to create this dataframe from your raw documents database. For instance the `pdf_converter` can create a `cdqa` dataframe from a directory containing `.pdf` files:

```python
from cdqa.utils.converters import pdf_converter

df = pdf_converter(directory_path='path_to_pdf_folder')
```

You will need to install [Java OpenJDK](https://openjdk.java.net/install/) to use this converter. We currently have converters for:

- pdf
- markdown

We plan to improve and add more converters in the future. Stay tuned!

### Downloading pre-trained models and data

You can download the models and data manually from the GitHub [releases](https://github.com/cdqa-suite/cdQA/releases) or use our download functions:

```python
from cdqa.utils.download import download_squad, download_model, download_bnpp_data

directory = 'path-to-directory'

# Downloading data
download_squad(dir=directory)
download_bnpp_data(dir=directory)

# Downloading pre-trained BERT fine-tuned on SQuAD 1.1
download_model('bert-squad_1.1', dir=directory)
```

### Training models

Fit the pipeline on your corpus using the pre-trained reader:

```python
import pandas as pd
from ast import literal_eval
from cdqa.pipeline import QAPipeline

df = pd.read_csv('your-custom-corpus-here.csv', converters={'paragraphs': literal_eval})

cdqa_pipeline = QAPipeline(reader='bert_qa_vCPU-sklearn.joblib')
cdqa_pipeline.fit_retriever(df=df)
```

If you want to fine-tune the reader on your custom SQuAD-like annotated dataset:

```python
cdqa_pipeline = QAPipeline(reader='bert_qa_vGPU-sklearn.joblib')
cdqa_pipeline.fit_reader('path-to-custom-squad-like-dataset.json')
```

Save the reader model after fine-tuning:
```python
cdqa_pipeline.dump_reader('path-to-save-bert-reader.joblib')
```
### Making predictions

To get the best prediction given an input query:

```python
cdqa_pipeline.predict(query='your question')
```

To get the N best predictions:
```python
cdqa_pipeline.predict(query='your question', n_predictions=N)
```

There is also the possibility to change the weight of the retriever score
versus the reader score in the computation of final ranking score (the default is 0.35, which is shown to be the best weight on the development set of SQuAD 1.1-open)

```python
cdqa_pipeline.predict(query='your question', retriever_score_weight=0.35)
```

### Evaluating models

In order to evaluate models on your custom dataset you will need to annotate it. The annotation process can be done in 3 steps:

1. Convert your pandas DataFrame into a json file with SQuAD format:

    ```python
    from cdqa.utils.converters import df2squad

    json_data = df2squad(df=df, squad_version='v1.1', output_dir='.', filename='dataset-name')
    ```

2. Use an annotator to add ground truth question-answer pairs:

    Please refer to our [`cdQA-annotator`](https://github.com/cdqa-suite/cdQA-annotator), a web-based annotator for closed-domain question answering datasets with SQuAD format.

3. Evaluate the pipeline object:

    ```python
    from cdqa.utils.evaluation import evaluate_pipeline

    evaluate_pipeline(cdqa_pipeline, 'path-to-annotated-dataset.json')

    ```

4. Evaluate the reader:

    ```python
    from cdqa.utils.evaluation import evaluate_reader

    evaluate_reader(cdqa_pipeline, 'path-to-annotated-dataset.json')
    ```

## Notebook Examples

We prepared some notebook examples under the [examples](examples) directory.

You can also play directly with these notebook examples using [Binder](https://gke.mybinder.org/) or [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb):

| Notebook                         | Hardware     | Platform                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1] First steps with cdQA        | CPU or GPU | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cdqa-suite/cdQA/master?filepath=examples%2Ftutorial-first-steps-cdqa.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdqa-suite/cdQA/blob/master/examples/tutorial-first-steps-cdqa.ipynb)   |
| [2] Using the PDF converter      | CPU or GPU | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cdqa-suite/cdQA/master?filepath=examples%2Ftutorial-use-pdf-converter.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdqa-suite/cdQA/blob/master/examples/tutorial-use-pdf-converter.ipynb) |
| [3] Training the reader on SQuAD | GPU        | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdqa-suite/cdQA/blob/master/examples/tutorial-train-reader-squad.ipynb)                                                                                                                                                         |

Binder and Google Colaboratory provide temporary environments and may be slow to start but we recommend them if you want to get started with `cdQA` easily.

## Deployment

### Manual

You can deploy a `cdQA` REST API by executing:

```shell
export dataset_path=path-to-dataset.csv
export reader_path=path-to-reader-model

FLASK_APP=api.py flask run -h 0.0.0.0
```

You can now make requests to test your API (here using [HTTPie](https://httpie.org/)):

```shell
http localhost:5000/api query=='your question here'
```

If you wish to serve a user interface on top of your `cdQA` system, follow the instructions of [cdQA-ui](https://github.com/cdqa-suite/cdQA-ui), a web interface developed for `cdQA`.

## Contributing

Read our [Contributing Guidelines](.github/CONTRIBUTING.md).

## References

| Type                 | Title                                                                                                                                        | Author                                                                                 | Year |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---- |
| :video_camera: Video | [Stanford CS224N: NLP with Deep Learning Lecture 10 – Question Answering](https://youtube.com/watch?v=yIdF-17HwSk)                           | Christopher Manning                                                                    | 2019 |
| :newspaper: Paper    | [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)                                                        | Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes                                   | 2017 |
| :newspaper: Paper    | [Neural Reading Comprehension and Beyond](https://cs.stanford.edu/people/danqi/papers/thesis.pdf)                                            | Danqi Chen                                                                             | 2018 |
| :newspaper: Paper    | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                         | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                           | 2018 |
| :newspaper: Paper    | [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006)                                               | Noah A. Smith                                                                          | 2019 |
| :newspaper: Paper    | [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718)                                                | Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, Jimmy Lin | 2019 |
| :newspaper: Paper    | [Data Augmentation for BERT Fine-Tuning in Open-Domain Question Answering](https://arxiv.org/abs/1904.06652)                                 | Wei Yang, Yuqing Xie, Luchen Tan, Kun Xiong, Ming Li, Jimmy Lin                        | 2019 |
| :newspaper: Paper    | [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)                                                                             | Rodrigo Nogueira, Kyunghyun Cho                                                        | 2019 |
| :newspaper: Paper    | [MRQA: Machine Reading for Question Answering](https://mrqa.github.io/)                                                                      | Jonathan Berant, Percy Liang, Luke Zettlemoyer                                         | 2019 |
| :newspaper: Paper    | [Unsupervised Question Answering by Cloze Translation](https://arxiv.org/abs/1906.04980)                                                     | Patrick Lewis, Ludovic Denoyer, Sebastian Riedel                                       | 2019 |
| :computer: Framework | [Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)                                           | Pedregosa et al.                                                                       | 2011 |
| :computer: Framework | [PyTorch](https://arxiv.org/abs/1906.04980)                                                                                                  | Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan                               | 2016 |
| :computer: Framework | [Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.](https://github.com/huggingface/transformers) | Hugging Face                                                                           | 2018 |

## LICENSE

[Apache-2.0](LICENSE)
