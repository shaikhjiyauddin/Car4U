FROM python:3.11-slim

WORKDIR /app

# System deps needed by pandas/scikit-learn wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models at build time if they don't already exist in the image.
# (In production you'd usually COPY pre-trained models/ in instead — see docker-compose.yml)
RUN python src/train_models.py && python src/train_predictor.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
