from flask import Flask, request, jsonify
from flask_cors import CORS

import os
from ast import literal_eval
import pandas as pd

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline

app = Flask(__name__)
CORS(app)

dataset_path = os.environ["dataset_path"]
reader_path = os.environ["reader_path"]

df = pd.read_csv(dataset_path, converters={"paragraphs": literal_eval})
df = filter_paragraphs(df)

cdqa_pipeline = QAPipeline(reader=reader_path)
cdqa_pipeline.fit_retriever(df=df)


@app.route("/api", methods=["GET"])
def api():

    query = request.args.get("query")
    prediction = cdqa_pipeline.predict(query=query)

    return jsonify(
        query=query, answer=prediction[0], title=prediction[1], paragraph=prediction[2]
    )
