# Model-service
Contains the wrapper service for the ML model

- Fetch a trained ML model from somewhere.
- Embed the ML model in a Flask webservice, so it can be queried via REST.
- Use the same pre-processing for the data that was used for training.
- The webservice is containerized and released on GitHub through a workflow.
- The image is versioned automatically, e.g., through release tags

## Instructions
### Helm-Chart Installation
To install the application as a helm chart, run the following command from the root directory of this repository:
- `helm install <model-service-name> .\model-service-chart\`

# Monitoring

We have introduced 5 domain-specific metrics in our webapp:

- counter_happy_predictions (counts how many reviews that are classified as positive)
- counter_sad_predictions (counts how many reviews that are classified as negative)
- gauge_time (measures the time for a specific step using a label, so far we measure the time for processing and prediction)
- summary_time (measures the same time as gauge_time, but by using the Summary metric we can easily measure the number of predictions made and the total time for different steps)
- histogram_size_of_input (calculates the size of the input and by using buckets we can easily visualize it)
