FROM python:3.9-slim

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./models/twt_roberta_model.pkl ./models/twt_roberta_model.pkl
COPY model_service_twt_roberta.py .

COPY ./models/random_forest_model.joblib ./models/random_forest_model.joblib
COPY model_service_random_forest.py .

COPY c1_BoW_Sentiment_Model.pkl .
COPY c2_Classifier_Sentiment_Model.joblib .
COPY model_service.py .


EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["model_service_random_forest.py"]
