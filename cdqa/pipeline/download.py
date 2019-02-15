import wget

squad_urls = [
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
]

for squad_url in squad_urls:
    wget.download(url=squad_url, out='data')

wget.download(url='https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py')
