# cdQA

[![Build Status](https://travis-ci.com/fmikaelian/cdQA.svg?token=Vzy9RRKRZ41ynd9q2BRX&branch=develop)](https://travis-ci.com/fmikaelian/cdQA) [![License](
https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

An end-to-end closed-domain question answering system with BERT and classic IR methods 📚

- [Installation](#installation)
- [Getting started](#getting-started)
  - [Preparing your data](#preparing-your-data)
  - [Training models](#training-models)
  - [Making predictions](#making-predictions)
  - [Practical examples](#practical-examples)
- [Contributing](#contributing)
- [References](#references)

## Installation

```shell
git clone https://github.com/fmikaelian/cdQA.git
pip install .
```

Note: Experiments have been done on an AWS EC2 `p3.2xlarge` Deep Learning AMI + a single Tesla V100 16GB with 16-bits training enabled (to accelerate training and prediction). To enable this feature, you will need to install [`apex`](https://github.com/nvidia/apex):

```shell
git clone https://github.com/NVIDIA/apex.git
cd apex/
python setup.py install --cuda_ext --cpp_ext
```

## Getting started

To download existing data and models automatically from the Github releases, you will need a personal Github token. You can find [how to create one here](https://github.com/settings/tokens) (you only need to select the `repo` scope).

```shell
export token='YOUR_GITHUB_TOKEN'
```

You can now execute the `download.py` to get all Github release assets:

```shell
python cdqa/pipeline/download.py
```

### Preparing your data

To be defined.

### Training models

You can train with a sklearn like api:

```python
from cdqa.pipeline.trainer import train
```

### Making predictions

```python
from cdqa.pipeline.predictor import predict
```

### Practical examples

This worfklow is described in our [`examples`](examples) notebook.

You can also use [`pipeline`](cdqa/pipeline) scripts direclty to use the application:

```
python cdqa/pipeline/train.py --data
python cdqa/pipeline/predict.py --data
```

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