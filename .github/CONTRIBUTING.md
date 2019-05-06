# Contributing

- [Contributions](#contributions)
- [Repository Structure](#repository-structure)

## Contributions

To contribute to this repository, you will need to follow the git branch workflow:

- Create a feature branch from `develop` branch with the name of the issue you want to fix.
- Commit in this new feature branch until your fix is done while referencing the issue number in your commit message.
- Open a pull request in order to merge you branch with the `develop` branch
- Discuss with peers and update your code until pull request is accepted by repository admins.
- Delete you feature branch.
- Synchonise your repository with the latest `develop` changes.
- Repeat!

See more about this workflow at https://guides.github.com/introduction/flow/

## Repository Structure

```
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
│   ├── tutorial-predict-pipeline.ipynb --> examples to use cdqa for prediction
├── logs --> stores the outpout predictions and metrics
├── models --> stores the trained models
├── requirements.txt
├── setup.py
└── tests --> unit tests for the application
    ├── __init__.py
    └── test_pipeline.py
```