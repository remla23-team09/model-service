"""
Flask API of the REMLA base_project model.
"""
#import traceback
import joblib
#import re
from flask import Flask, jsonify, request, Response
from flasgger import Swagger
#import pandas as pd
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
import pickle
import time
import sys
from prometheus_client import Counter, Gauge, Histogram, generate_latest

app = Flask(__name__)
swagger = Swagger(app)

happy_predictions = Counter("counter_happy_predictions", "Count the number of happy faces.")
sad_predictions = Counter("counter_sad_predictions", "Count the number of sad faces.")
prediction_time = Gauge("gauge_prediction_time", "Count the duration for the prediction for different steps.", ["step"])
size_of_input = Histogram("histogram_size_of_input", "The number of characters in the input.", buckets=[0, 5, 10, 15, 25, 50, 75, 100])

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

    global happy_predictions, sad_predictions, prediction_time, size_of_input

    # Process data
    start_time_processing = time.time()
    input_data = request.get_json()
    text = input_data.get('text')
    processed_text = prepare(text)
    end_time_processing = time.time()

    # Load model and predict
    start_time_prediction = time.time()
    model = joblib.load('c2_Classifier_Sentiment_Model.joblib')
    prediction = model.predict(processed_text)[0]
    end_time_prediction = time.time()
    
    res = {
        "prediction": str(prediction),
        "text": text
    }

    # Update counters
    if prediction == 1:
        happy_predictions.inc()
    else:
        sad_predictions.inc()
    
    # Update gauge
    elapsed_time_processing= end_time_processing - start_time_processing
    prediction_time.labels("processing").set(elapsed_time_processing)
    elapsed_time_prediction= end_time_prediction - start_time_prediction
    prediction_time.labels("prediction").set(elapsed_time_prediction)

    # Update histogram
    size_of_input.observe(len(text))

    return jsonify(res)

@app.route('/metrics', methods=['GET'])
def metrics():

    return Response(generate_latest(), mimetype="text/plain")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)