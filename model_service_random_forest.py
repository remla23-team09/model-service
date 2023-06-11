import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pickle
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
swagger = Swagger(app)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

happy_predictions = Counter("counter_happy_predictions", "Count the number of happy faces.")
sad_predictions = Counter("counter_sad_predictions", "Count the number of sad faces.")
time_individual = Gauge("gauge_time", "Count the duration for different steps.", ["step"])
size_of_input = Histogram("histogram_size_of_input", "The number of characters in the input.", buckets=[0, 5, 10, 15, 20, 25, 50, 75, 100])
time_summary = Summary("summary_time", "Summarizing duration for different steps", ["step"])

model = None

with open('./models/random_forest_model.joblib', 'rb') as model_file:   
        print('loading model') 
        model = joblib.load(model_file)
        print('model loaded')

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

    global happy_predictions, sad_predictions, time_individual, time_summary, size_of_input
    global model

    # Process data
    start_time_processing = time.time()
    input_data = request.get_json()
    text = input_data.get('text')
    processed_text = prepare(text)
    end_time_processing = time.time()

    # Load model and predict
    start_time_prediction = time.time()
    sentiment = model.predict(processed_text)[0]
    end_time_prediction = time.time()
    
    res = {
        "sentiment": str(sentiment),
        "text": text
    }

    # Update counters
    if sentiment == 1:
        happy_predictions.inc()
    else:
        sad_predictions.inc()

    # Update gauge
    elapsed_time_processing= end_time_processing - start_time_processing
    elapsed_time_prediction= end_time_prediction - start_time_prediction
    time_individual.labels("processing").set(elapsed_time_processing)
    time_individual.labels("sentiment").set(elapsed_time_prediction)

    # Update histogram
    size_of_input.observe(len(text))

    # Update summary
    time_summary.labels("processing").observe(elapsed_time_processing)
    time_summary.labels("sentiment").observe(elapsed_time_prediction)

    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)