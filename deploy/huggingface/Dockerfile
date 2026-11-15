# Public deployment on Hugging Face Spaces: one container, models baked in, embedded Qdrant
# (QDRANT_URL=:memory: -> the app loads models/items.parquet into an in-memory collection at startup).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn==0.34.0 numpy==2.0.2 pandas==2.3.3 pyarrow==21.0.0 \
    lightgbm==4.7.0 qdrant-client==1.20.0
COPY src /app/src
COPY app /app/app
COPY models /app/models
ENV MODELS_DIR=/app/models QDRANT_URL=:memory: QDRANT_COLLECTION=games
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
