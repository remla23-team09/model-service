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
import time
from prometheus_client import start_http_server, Histogram, Summary

app = Flask(__name__)
swagger = Swagger(app)

elapsedPredictionTime = 0
countHappyPredictions = 0
countSadPredictions = 0

request_latency_histogram = Histogram(
    'elapsedTime',
    'Latency of web app requests',
    ['endpoint']
)

elapsed_prediction_summary = Summary(
    'elapsedPredictionTime',
    'Elapsed time during prediction'
)

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

    global countHappyPredictions, countSadPredictions, elapsedPredictionTime, request_latency_histogram, elapsed_prediction_summary

    with request_latency_histogram.labels(endpoint='/processing').time():
        startTime = time.time()
        input_data = request.get_json()
        text = input_data.get('text')
        processed_text = prepare(text)

    with request_latency_histogram.labels(endpoint='/loading').time(): 
        model = joblib.load('c2_Classifier_Sentiment_Model.joblib')

    with request_latency_histogram.labels(endpoint='/prediction').time():
        prediction = model.predict(processed_text)[0]
        endTime = time.time()
    
    res = {
        "prediction": str(prediction),
        "text": text
    }

    if prediction == 1:
        countHappyPredictions += 1
    else:
        countSadPredictions += 1
    elapsedPredictionTime = endTime - startTime
    elapsed_prediction_summary.observe(elapsedPredictionTime)

    return jsonify(res)

@app.route('/metrics', methods=['GET'])
def metrics():
    #global countHappyPredictions, countSadPredictions, elapsedPredictionTime, request_latency_histogram, elapsed_prediction_summary
    
    m = "# HELP num_predictions Number of predictions\n"
    m += "# TYPE num_predictions counter\n"
    m += "# HELP num_happy_predictions Number of happy predictions\n"
    m += "# TYPE num_happy_predictions counter\n"
    m += "# HELP num_sad_predictions Number of sad predictions\n"
    m += "# TYPE num_sad_predictions counter\n"

    #m+= "num_predictions{{page=\"sub\"}} {}\n".format(countPredictions)
    m+= "num_happy_predictions{{page=\"sub\"}} {}\n".format(countHappyPredictions)
    m+= "num_sad_predictions{{page=\"sub\"}} {}\n".format(countSadPredictions)
    m+= "elapsed_time{{page=\"sub\"}} {}\n".format(elapsedPredictionTime)

    return Response(m, mimetype="text/plain")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)