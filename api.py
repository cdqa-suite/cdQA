from flask import Flask, request, jsonify

import os
from ast import literal_eval
import pandas as pd

from cdqa.utils.converter import filter_paragraphs
from cdqa.reader.bertqa_sklearn import BertQA
from cdqa.pipeline.qa_pipeline import QAPipeline

app = Flask(__name__)

df = pd.read_csv('../data/bnpp_newsroom_v1.0/bnpp_newsroom-v1.0.csv', converters={'paragraphs': literal_eval})
df['paragraphs'] = df['paragraphs'].apply(filter_paragraphs)
df['content'] = df['paragraphs'].apply(lambda x: ' '.join(x))

qa_pipe = QAPipeline(model = '../models/bert_qa_squad_vCPU/bert_qa_squad_vCPU-sklearn.joblib')
qa_pipe.fit(df)

@app.route('/api', methods=['GET'])
def api():
    query = request.query_string
    prediction = qa_pipe.answer(query)
    return jsonify(query=query,
                   prediction=prediction)