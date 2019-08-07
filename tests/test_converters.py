import os
import wget
from cdqa.utils.converters import pdf_converter, md_converter

def download_test_assets(type):
  directory = 'data/{}/'.format(type)
  if type == 'pdf':
    assets_urls = [
        'https://invest.bnpparibas.com/documents/1q19-pr-12648',
        'https://invest.bnpparibas.com/documents/4q18-pr-18000',
        'https://invest.bnpparibas.com/documents/4q17-pr'
    ]
  elif type == 'md':
    assets_urls = [
        'https://raw.githubusercontent.com/cdqa-suite/cdQA/master/README.md',
        'https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/docs/source/quickstart.md',
        'https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/docs/source/migration.md'
    ]
  print('\nDownloading {} assets...'.format(type))

  if not os.path.exists(directory):
      os.makedirs(directory)
  for url in assets_urls:
      wget.download(url=url, out=directory)

def test_converter(converter, type):
    download_test_assets(type)
    df = converter(directory_path='data/{}/'.format(type))
    errors = []

    # replace assertions by conditions
    if not df.shape == (3, 2):
        errors.append('resulting dataframe has unexpected shape.')
    if not (isinstance(df.paragraphs[0][0], str) and isinstance(df.title[0], str)):
        errors.append('paragraph column content has wrong format.')

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_pdf_converter():
    test_converter(pdf_converter, type='pdf')

def test_md_converter():
    test_converter(md_converter, type='md')