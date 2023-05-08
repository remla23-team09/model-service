"""
Flask API of the REMLA base_project model.
"""
#import traceback
import joblib
import re
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)
swagger = Swagger(app)

def prepare(text):
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    processed_input = cv.transform([text]).toarray()[0]
    return [processed_input]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the sentiment of provided text.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: text to be classified.
          required: True
          schema:
            type: object
            required: text
            properties:
                text:
                    type: string
                    example: I'm feeling happy today.
    responses:
      200:
        description: "The result of the classification: '0' or '1'."
    """

    input_data = request.get_json()
    text = input_data.get('text')
    processed_text = prepare(text)

    model = joblib.load('c2_Classifier_Sentiment_Model.joblib')

    prediction = model.predict(processed_text)[0]

    res = {
        "prediction": str(prediction),
        "text": text
    }

    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)