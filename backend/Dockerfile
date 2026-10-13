# Build from the repository root:  docker build -f backend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Small hosts give a fraction of one CPU; extra BLAS threads only add contention.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /srv

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend backend
# The bundled synthetic dataset is seeded into the database on startup.
COPY data/samples data/samples

RUN useradd --create-home appuser
USER appuser

WORKDIR /srv/backend
EXPOSE 8000

# Hosts such as Render inject PORT; default to 8000 elsewhere.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
