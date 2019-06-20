import os
import re
import sys
from tika import parser
import pandas as pd


def pdf_converter(directory_path):
    list_pdf = os.listdir(directory_path)
    df = pd.DataFrame(columns=['title', 'paragraphs'])
    for i, pdf in enumerate(list_pdf):
        df.loc[i] = [pdf, None]
        raw = parser.from_file(directory_path+pdf)
        s = raw['content']
        paragraphs = re.split(u'\n(?=\u2028|[A-Z-0-9])', s)
        list_par = []
        for p in paragraphs:
            if len(p) >= 200:
                list_par.append(p.replace('\n', ''))
            df.loc[i, 'paragraphs'] = list_par
    return df


def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf_converter.py <path-to-folder-with-pdf-files>",
              "<csv-output-path>")
    else:
        data = pdf_converter(sys.argv[1])
        data.to_csv(sys.argv[2])


if __name__ == '__main__':
    main()
