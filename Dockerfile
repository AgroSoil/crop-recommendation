FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-setuptools \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install scikit-learn==1.5.2

COPY app.py app.py
COPY crops_info.json crops_info.json
COPY random_forest_model.pkl random_forest_model.pkl
COPY label_encoder.pkl label_encoder.pkl
COPY scaler.pkl scaler.pkl  
COPY templates/ templates/
COPY static/ static/

# flask port
EXPOSE 5008

# command to run application
CMD ["python", "app.py"]
