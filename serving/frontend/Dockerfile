FROM python:3.10-slim

WORKDIR /app

COPY proxy_server.py /app/proxy_server.py
COPY templates /app/templates

RUN pip install --no-cache-dir \
    Flask \
    requests

EXPOSE 5005

ENTRYPOINT ["python", "proxy_server.py"]
