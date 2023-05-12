"""
Flask API of the REMLA base_project model.
"""
#import traceback
import joblib
import re
from flask import Flask, jsonify, request, Response
from flasgger import Swagger
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)
swagger = Swagger(app)


countTextPreprocess = 0
countPredictions = 0
countHappyPredictions = 0

def prepare(text):
    global countTextPreprocess
    countTextPreprocess += 1

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

    global countPredictions, countHappyPredictions
    countPredictions += 1

    input_data = request.get_json()
    text = input_data.get('text')
    processed_text = prepare(text)

    model = joblib.load('c2_Classifier_Sentiment_Model.joblib')

    prediction = model.predict(processed_text)[0]

    res = {
        "prediction": str(prediction),
        "text": text
    }

    if prediction == 1:
        countHappyPredictions += 1

    return jsonify(res)

@app.route('/metrics', methods=['GET'])
def metrics():
    global countTextPreprocess, countPredictions, countHappyPredictions
    
    m = "# HELP num_text_preprocess Number of text preprocess\n"
    m += "# TYPE num_text_preprocess counter\n"
    m += "# HELP num_predictions Number of predictions\n"
    m += "# TYPE num_predictions counter\n"
    m += "# HELP num_happy_predictions Number of happy predictions\n"
    m += "# TYPE num_happy_predictions counter\n"

    m+= "num_text_preprocess{{page=\"index\"}} {}\n".format(countTextPreprocess)
    m+= "num_predictions{{page=\"sub\"}} {}\n".format(countPredictions)
    m+= "num_happy_predictions{{page=\"sub\"}} {}\n".format(countHappyPredictions)

    return Response(m, mimetype="text/plain")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)