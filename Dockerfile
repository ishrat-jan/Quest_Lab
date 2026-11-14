# CPU-only inference image. Render (or any host) sets $PORT; defaults to 8000 locally.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# torch from the CPU wheel index (the default PyPI wheel bundles CUDA and is several GB)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs configs
COPY src src
COPY frontend frontend
# trained checkpoint, held-out test samples and real metrics (produced by training)
COPY artifacts artifacts

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn pde_service.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
