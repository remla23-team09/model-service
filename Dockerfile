FROM python:3.9-slim

WORKDIR /root

COPY requirements.txt /root/
RUN pip install -r requirements.txt

COPY c1_BoW_Sentiment_Model.pkl /root/
COPY c2_Classifier_Sentiment_Model.joblib /root/
COPY model_service.py /root/

COPY twt_roberta_sentiment_model.pkl /root/
COPY model_service_twt_roberta.py /root/

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["model_service_twt_roberta.py"]
