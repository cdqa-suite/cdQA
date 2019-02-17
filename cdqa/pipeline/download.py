import os
import wget
import requests
from github import Github

def main():
    squad_urls = [
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    ]

    for squad_url in squad_urls:
        wget.download(url=squad_url, out='data')

    wget.download(url='https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py')

    token = os.environ['token']
    g = Github(token)

    repo = g.get_repo('fmikaelian/cdQA')
    release = repo.get_release('bert_qa_squad_v1.1')
    assets = release.get_assets()

    headers = {
        'Authorization': 'token {}'.format(token),
        'Accept': 'application/octet-stream'
    }

    for asset in assets:
        print(asset.name, asset.url)
        if (os.path.splitext(asset.name)[1] == '.bin') or (asset.name == 'bert_config.json'):
            directory = 'models'
        else:
            directory = 'logs'
        response = requests.get(asset.url, headers=headers)
        if not os.path.exists(os.path.join(directory, release.tag_name)):
            os.makedirs(os.path.join(directory, release.tag_name))
        with open(os.path.join(directory, release.tag_name, asset.name), 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)

if __name__ == '__main__':
    main()