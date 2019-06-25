import os
import wget


def download_squad_assets():
    directory = 'data/SQuAD_1.1'
    squad_urls = [
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    ]

    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Downloading SQuAD v1.1 data...")

    for squad_url in squad_urls:
        wget.download(url=squad_url, out=directory)

def download_models():
    directory = 'models'
    models_url = [
    'https://github.com/cdqa-suite/cdQA/releases/download/bert_qa_vGPU/bert_qa_vGPU-sklearn.joblib',
    'https://github.com/cdqa-suite/cdQA/releases/download/bert_qa_vCPU/bert_qa_vCPU-sklearn.joblib'
    ]

    print('\nDownloading trained models...')

    if not os.path.exists(directory):
        os.makedirs(directory)
    for url in models_url:
        wget.download(url=url, out=directory)

def download_bnp_data():
    directory = 'data/bnpp_newsroom_v1.1'
    url = 'https://github.com/cdqa-suite/cdQA/releases/download/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv'

    print("\nDownloading BNP data...")

    if not os.path.exists(directory):
        os.makedirs(directory)

    wget.download(url=url, out=directory)

if __name__ == '__main__':
    directory = 'data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    download_squad_assets()
    download_models()
    download_bnp_data()
