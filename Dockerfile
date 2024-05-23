FROM python:3.8-slim

WORKDIR /app

COPY perceptron.py .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python3 perceptron.py

ENTRYPOINT ["python3"]
CMD ["perceptron.py"]