FROM python:3.9-slim as packages

WORKDIR /root
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM packages as data
COPY ./data/ ./data/

FROM data as models
COPY ./models/ ./models/

FROM models as app
COPY ./src/ ./src/
EXPOSE 8080

WORKDIR /root/src/
CMD ["python", "interface.py"]