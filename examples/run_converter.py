import pandas as pd
from cdqa.utils.converter import df2squad
from ast import literal_eval

# https://stackoverflow.com/questions/32742976/how-to-read-a-column-of-csv-as-dtype-list-using-pandas
df = pd.read_csv('bnpp_newsroom_v1.0.csv', converters={'paragraphs': literal_eval})

json_data = df2squad(df=df, version='v2.0', output_dir='./')
