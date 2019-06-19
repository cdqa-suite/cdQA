from flask import Flask, request, jsonify
from flask_cors import CORS

import os
from ast import literal_eval
import pandas as pd

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline.cdqa_sklearn import QAPipeline

app = Flask(__name__)
CORS(app)

df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = filter_paragraphs(df)

cdqa_pipeline = QAPipeline(reader='models/bert_qa_squad_v1.1_sklearn/bert_qa_squad_v1.1_sklearn.joblib')
cdqa_pipeline.fit(X=df)

@app.route('/api', methods=['GET'])
def api():
    
    query = request.args.get('query')
    prediction = cdqa_pipeline.predict(X=query)

    return jsonify(query=query,
                   answer=prediction[0],
                   title=prediction[1],
                   paragraph=prediction[2])
