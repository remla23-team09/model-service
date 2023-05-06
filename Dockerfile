FROM python:3.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY BoW_Sentiment_Model.pkl /root/
COPY Classifier_Sentiment_Model /root/
COPY model_service.py /root/
ENTRYPOINT ["python"]
CMD ["model_service.py"]
