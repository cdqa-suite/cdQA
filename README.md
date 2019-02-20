# cdQA

[![Build Status](https://travis-ci.com/fmikaelian/cdQA.svg?token=Vzy9RRKRZ41ynd9q2BRX&branch=develop)](https://travis-ci.com/fmikaelian/cdQA) [![License](
https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

An end-to-end closed-domain question answering system with BERT and classic IR methods 📚

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Getting started](#getting-started)
- [Contributing](#contributing)
- [References](#references)

## Installation

Note: Experiments have been done on an AWS EC2 `p3.2xlarge` Deep Learning AMI with a single Tesla V100 16GB.

```shell
git clone https://github.com/fmikaelian/cdQA.git
pip install .
```

## Repository Structure

```shell
├── LICENSE
├── README.md
├── cdqa
│   ├── __init__.py
│   ├── pipeline --> all steps as scripts to use the application
│   │   ├── __init__.py
│   │   ├── download.py --> downloads all assets needed to use the application (data, models)
│   │   ├── predict.py --> performs a prediction given a sample
│   │   ├── processing.py --> processes raw data to a format usable for model training
│   │   └── train.py --> trains a model given a input dataset already processed
│   ├── reader
│   │   ├── __init__.py
│   │   └── run_squad.py --> a miror of pytorch-pretrained-BERT example (used for pipeline steps)
│   ├── retriever
│   │   ├── __init__.py
│   │   └── tfidf_doc_ranker.py --> the logic for the document retriever
│   ├── scrapper
│   │   ├── __init__.py
│   │   └── bs4_scrapper.py --> the logic for the dataset scrapper
│   └── utils --> utility functions used in the pipeline (to avoid flooding pipeline scripts)
│       ├── __init__.py
│       └── converter.py --> the logic for converting the dataset to SQuAD format
├── data --> the raw datasets
├── examples --> examples to use different parts of the appplication
│   ├── run_converter.py
│   └── run_retriever.py
├── logs --> stores the outpout predictions and metrics
├── models --> stores the trained models
├── requirements.txt
├── samples --> sample data for tests or examples
├── setup.py
└── tests --> unit tests for the application
    ├── __init__.py
    └── test_pipeline.py
```

## Getting started

Download existing data and models with the `download.py` script:

```shell
export token='YOUR_GITHUB_TOKEN'
python cdqa/pipeline/download.py
```

You can now execute the [`examples`](examples) or the [`pipeline`](cdqa/pipeline) steps to use the application.

## Contributing

To contribute to this repository, you will need to follow the git branch workflow:

- Create a feature branch from `develop` branch with the name of the issue you want to fix.
- Commit in this new feature branch until your fix is done while referencing the issue number in your commit message.
- Open a pull request in order to merge you branch with the `develop` branch
- Discuss with peers and update your code until pull request is accepted by repository admins.
- Delete you feature branch.
- Synchonise your repository with the latest `develop` changes.
- Repeat!

See more about this workflow in: https://guides.github.com/introduction/flow/

## References

| Type              | Title                                                                                                                | Author                                                                                 | Year |
|-------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|------|
| :newspaper: Paper | [End-to-End Open-Domain Question Answering with BERTserini]([link](https://arxiv.org/abs/1902.01718))                | Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, Jimmy Lin | 2019 |
| :newspaper: Paper | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                           | 2018 |
| :newspaper: Paper | [Neural Reading Comprehension and Beyond](https://cs.stanford.edu/people/danqi/papers/thesis.pdf)                    | Danqi Chen                                                                             | 2018 |