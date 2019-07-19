import os
import wget
from cdqa.utils.converters import pdf_converter

def download_pdf():
  directory = 'data/pdf/'
  models_url = [
      'https://invest.bnpparibas.com/documents/1q19-pr-12648',
      'https://invest.bnpparibas.com/documents/4q18-pr-18000',
      'https://invest.bnpparibas.com/documents/4q17-pr'
  ]

  print('\nDownloading PDF files...')

  if not os.path.exists(directory):
      os.makedirs(directory)
  for url in models_url:
      wget.download(url=url, out=directory)

def test_pdf_converter():
    download_pdf()
    df = pdf_converter(directory_path='data/pdf/')
    errors = []

    # replace assertions by conditions
    if not df.shape == (3, 2):
        errors.append('resulting dataframe has unexpected shape.')
    if not type(df.paragraphs[0][0]) == str and type(df.paragraphs[0]) == str:
        errors.append('paragraph column content has wrong format.')

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))