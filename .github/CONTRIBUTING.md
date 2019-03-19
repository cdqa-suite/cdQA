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
│   │   ├── bertqa_sklearn.py --> A BertForQuestionAnswering sklearn wrapper based on run_squad.py's main() function
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