import pandas as pd
from cdqa.utils.converter import df2squad, filter_paragraphs
from ast import literal_eval

# https://stackoverflow.com/questions/32742976/how-to-read-a-column-of-csv-as-dtype-list-using-pandas
df = pd.read_csv('data/bnpp_newsroom_v1.0/bnpp_newsroom_v1.0.csv', converters={'paragraphs': literal_eval})

df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)

json_data = df2squad(df=df, filename='bnpp_newsroom-v1.0', squad_version='v2.0', output_dir='data')
