import pandas as pd
from reading-comprehension.utils.converter import df2squad

df = pd.read_csv('data.csv')

json_data = df2squad(df=df, version='v2.0', output_dir='./')