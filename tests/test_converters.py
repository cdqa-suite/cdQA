import os
import wget
from cdqa.utils.converters import pdf_converter

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
        '',
        '',
        ''
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
    if not type(df.paragraphs[0][0]) == str and type(df.paragraphs[0]) == str:
        errors.append('paragraph column content has wrong format.')

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_pdf_converter():
    test_converter(pdf_converter, type='pdf')
