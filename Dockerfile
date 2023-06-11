FROM python:3.9-slim

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY c1_BoW_Sentiment_Model.pkl .
COPY c2_Classifier_Sentiment_Model.joblib .
COPY model_service.py .

COPY ./models/ ./models/
COPY model_service_twt_roberta.py .

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["model_service_twt_roberta.py"]
