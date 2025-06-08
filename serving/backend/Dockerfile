FROM python:3.10-slim

WORKDIR /mlflow-server

COPY serving/backend/lung_cancer_model /mlflow-server/lung_cancer_model
COPY serving/backend/mlruns /mlflow-server/mlruns
COPY lung_cancer_detection /mlflow-server/lung_cancer_detection

RUN pip install --no-cache-dir \
    mlflow \
    torch \
    torchvision \
    pytorch-lightning

EXPOSE 8888

CMD ["mlflow", "models", "serve", "-m", "/mlflow-server/lung_cancer_model", "-p", "8888", "--host", "0.0.0.0", "--no-conda"]
