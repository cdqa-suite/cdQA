import os
import wget
import requests
from github import Github


def download_squad_assets():
    squad_urls = [
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    ]

    for squad_url in squad_urls:
        wget.download(url=squad_url, out='data')

    wget.download(
        url='https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py')


def download_releases_assets():
    token = os.environ['token']
    g = Github(token)

    repo = g.get_repo('fmikaelian/cdQA')

    headers = {
        'Authorization': 'token {}'.format(token),
        'Accept': 'application/octet-stream'
    }

    # download models
    models = ['bert_qa_squad_v1.1',
              'bert_qa_squad_v1.1_dev',
              'bert_qa_squad_v1.1_sklearn'
              ]
    
    for model in models:
        release = repo.get_release(model)
        assets = release.get_assets()

        for asset in assets:
            print(asset.name, asset.url)
            if (os.path.splitext(asset.name)[1] == '.bin') or (os.path.splitext(asset.name)[1] == '.joblib') or (asset.name == 'bert_config.json'):
                directory = 'models'
            else:
                directory = 'logs'
            response = requests.get(asset.url, headers=headers)
            if not os.path.exists(os.path.join(directory, release.tag_name)):
                os.makedirs(os.path.join(directory, release.tag_name))
            with open(os.path.join(directory, release.tag_name, asset.name), 'wb') as handle:
                for block in response.iter_content(1024):
                    handle.write(block)

    # download datasets
    release = repo.get_release('bnpp_newsroom_v1.0')
    assets = release.get_assets()

    for asset in assets:
        print(asset.name, asset.url)
        response = requests.get(asset.url, headers=headers)
        directory = 'data'
        if not os.path.exists(os.path.join(directory, release.tag_name)):
            os.makedirs(os.path.join(directory, release.tag_name))
        with open(os.path.join(directory, release.tag_name, asset.name), 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)


if __name__ == '__main__':
    download_squad_assets()
    download_releases_assets()
