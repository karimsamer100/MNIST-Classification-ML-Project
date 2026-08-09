FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MNIST-data/mnist_train.csv ./MNIST-data/mnist_train.csv
COPY MNIST-data/mnist_test.csv ./MNIST-data/mnist_test.csv
COPY src/ ./src/

CMD ["python", "src/phase1/main.py"]
