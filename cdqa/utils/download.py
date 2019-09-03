import os
import wget


def download_squad(dir="."):
    """
    Download SQuAD 1.1 and SQuAD 2.0 datasets

    Parameters
    ----------
    dir: str
        Directory where the dataset will be stored

    """

    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Download SQuAD 1.1
    print("Downloading SQuAD v1.1 data...")

    dir_squad11 = os.path.join(dir, "SQuAD_1.1")
    squad11_urls = [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    ]

    if not os.path.exists(dir_squad11):
        os.makedirs(dir_squad11)

    for squad_url in squad11_urls:
        file = squad_url.split("/")[-1]
        if os.path.exists(os.path.join(dir_squad11, file)):
            print(file, "already downloaded")
        else:
            wget.download(url=squad_url, out=dir_squad11)

    # Download SQuAD 2.0
    print("\nDownloading SQuAD v2.0 data...")

    dir_squad20 = os.path.join(dir, "SQuAD_2.0")
    squad20_urls = [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    ]

    if not os.path.exists(dir_squad20):
        os.makedirs(dir_squad20)

    for squad_url in squad20_urls:
        file = squad_url.split("/")[-1]
        if os.path.exists(os.path.join(dir_squad20, file)):
            print(file, "already downloaded")
        else:
            wget.download(url=squad_url, out=dir_squad20)


def download_model(model="bert-squad_1.1", dir="."):
    """
    Download pretrained models

    Parameters
    ----------
    model: str
        Model to be download. It should be one of the models in the list:
        'bert-squad1.1'

    dir: str
        Directory where the dataset will be stored

    """

    models_url = {
        "bert-squad_1.1": "https://github.com/cdqa-suite/cdQA/releases/download/bert_qa_vCPU/bert_qa_vCPU-sklearn.joblib"
    }

    if not model in models_url:
        print(
            "The model you chose does not exist. Please choose one of the following models:"
        )
        for model in models_url.keys():
            print(model)
    else:
        print("\nDownloading trained model...")

        dir = os.path.expanduser(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        url = models_url[model]
        file = url.split("/")[-1]
        if os.path.exists(os.path.join(dir, file)):
            print(file, "already downloaded")
        else:
            wget.download(url=url, out=dir)


def download_bnpp_data(dir="."):
    """
    Download BNP Paribas' dataset

    Parameters
    ----------
    dir: str
        Directory where the dataset will be stored

    """

    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    url = "https://github.com/cdqa-suite/cdQA/releases/download/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv"

    print("\nDownloading BNP data...")

    file = url.split("/")[-1]
    if os.path.exists(os.path.join(dir, file)):
        print(file, "already downloaded")
    else:
        wget.download(url=url, out=dir)
