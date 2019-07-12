# Contributing

Thank you for considering contributing to cdQA 🙏

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Table of Contents <!-- omit in toc -->
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Your First Contribution](#your-first-contribution)
  - [Unsure where to begin contributing to cdQA?](#unsure-where-to-begin-contributing-to-cdqa)
  - [Working on your first Pull Request?](#working-on-your-first-pull-request)
  - [Having trouble understanding the repository structure?](#having-trouble-understanding-the-repository-structure)

## Code of Conduct

This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code.

## How Can I Contribute?

cdQA is an open source project and we love to receive contributions from our community! There are many ways to contribute: improving the documentation, reporting bugs, suggesting or proposing new features, sharing your results...

## Your First Contribution

### Unsure where to begin contributing to cdQA?

You can start by looking through these beginner and help-wanted issues:

- Beginner issues - issues which should only require a few lines of code, and a test or two.
- Help wanted issues - issues which should be a bit more involved than beginner issues.

### Working on your first Pull Request?

Here are some example steps to get it done:

- Create a feature branch from `develop` branch with the name of the issue you want to fix.
- Commit in this new feature branch until your fix is done while referencing the issue number in your commit message.
- Open a pull request in order to merge you branch with the `develop` branch.
- Discuss with peers and update your code until pull request is accepted by repository admins.
- Delete you feature branch.
- Synchonise your repository with the latest `develop` changes.
- Repeat!

See more about this workflow at https://guides.github.com/introduction/flow/

### Having trouble understanding the repository structure?

```shell
├── Dockerfile
├── LICENSE
├── README.md
├── api.py --> the REST API of cdqa pipeline
├── cdqa
│   ├── __init__.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── cdqa_sklearn.py --> a full cdqa pipeline sklearn wrapper based on run_squad.py's main() function
│   ├── reader
│   │   ├── __init__.py
│   │   ├── bertqa_sklearn.py --> a BertForQuestionAnswering sklearn wrapper based on run_squad.py's main() function
│   │   └── run_squad.py --> a miror of pytorch-pretrained-BERT example (used for pipeline steps)
│   ├── retriever
│   │   ├── __init__.py
│   │   └── tfidf_sklearn.py --> the logic for the document retriever as a sklearn wrapper
│   ├── scrapper
│   │   ├── __init__.py
│   │   └── bs4_bnpp_newsroom.py --> the logic for the dataset scrapper
│   └── utils
│       ├── __init__.py
│       ├── converter.py --> the logic for converting the dataset to SQuAD format
│       └── download.py --> downloads all assets needed to use the application (data, models)
│       ├── filters.py
│       └── metrics.py
├── data
├── docs
│   └── latex
│       ├── cdqa.tex --> the research paper describing cdqa
│       └── neurips_2019.sty --> the style file for the research paper from neurips
├── examples
│   ├── tutorial-first-steps-cdqa.ipynb --> examples to use cdqa for prediction
├── logs --> stores the outpout predictions and metrics
├── models --> stores the trained models
├── requirements.txt
├── setup.py
└── tests --> unit tests for the application
    ├── __init__.py
    └── test_pipeline.py
```