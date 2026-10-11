# Build from the repository root:  docker build -f frontend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install -r frontend/requirements.txt

COPY .streamlit .streamlit
COPY frontend frontend
# Sample files offered as downloads in the sidebar.
COPY data/samples data/samples

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
