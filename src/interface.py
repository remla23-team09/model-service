import os
import importlib
from flask import Flask, jsonify, request
from flasgger import Swagger
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
swagger = Swagger(app)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

happy_predictions = Counter("counter_happy_predictions", "Count the number of happy faces.")
neutral_predictions = Counter("counter_neutral_predictions", "Count the number of neutral predictions.")
sad_predictions = Counter("counter_sad_predictions", "Count the number of sad faces.")
time_individual = Gauge("gauge_time", "Count the duration for different steps.", ["step"])
size_of_input = Histogram("histogram_size_of_input", "The number of characters in the input.", buckets=[0, 5, 10, 15, 20, 25, 50, 75, 100])
time_summary = Summary("summary_time", "Summarizing duration for different steps", ["step"])
load_model_time = Gauge("gauge_load_model_time", "Count the duration for loading the model.", ["model_name"])

model = None
model_module = None

@app.before_first_request
def load_model():
    global model,model_module

    model_name = os.environ.get('MODEL_NAME', 'base_model')  # Default to 'model1' if not specified
    model_module = importlib.import_module(f"models.{model_name}")
    
    start_time_model = time.time()
    if model_name == 'base_model':
        app.logger.info('loading base model...')
        model = model_module.init()
    elif model_name == 'random_forest':
        app.logger.info('loading random forest model...')
        model = model_module.init()
    elif model_name == 'twt_roberta':
        app.logger.info('loading twt roberta model...')
        model = model_module.init()
    
    time_elapsed = time.time() - start_time_model
    app.logger.info('model loaded in {} seconds'.format(time_elapsed))
    load_model_time.labels("{}".format(model_name)).set(time_elapsed)


@app.route('/predict', methods=['POST'])
def predict():
    global happy_predictions, sad_predictions, time_individual, time_summary, size_of_input
    global model

    app.logger.info("received request for random forest model...")

    # Process data
    start_time_processing = time.time()
    input_data = request.get_json()
    text = input_data.get('text')

    app.logger.info("input text: {}".format(text))

    processed_text = model_module.prepare(text)
    end_time_processing = time.time()

    # Load model and predict
    start_time_prediction = time.time()
    app.logger.info("predicting...")

    sentiment = model_module.predict_sentiment(model, processed_text)
    end_time_prediction = time.time()
    
    res = {
        "sentiment": str(sentiment),
        "text": text
    }

    # Update counters
    if sentiment == 2:
        happy_predictions.inc()
    elif sentiment == 1:
        neutral_predictions.inc()
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
    app.logger.info("prediction done!")
    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)