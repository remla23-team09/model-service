# model-service
Contains the wrapper service for the ML model

- Fetch a trained ML model from somewhere.
- Embed the ML model in a Flask webservice, so it can be queried via REST.
- Use the same pre-processing for the data that was used for training.
- The webservice is containerized and released on GitHub through a workflow.
- The image is versioned automatically, e.g., through release tags