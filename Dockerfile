# Self-contained image: corpus + index are built at image build time, so the
# container starts in seconds and needs only OPENAI_API_KEY at runtime.
#
#   docker build -t librarian .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=... librarian
#
# Works as-is on Hugging Face Spaces (Docker SDK, set PORT=7860), Render, Fly.io.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
RUN bash scripts/fetch_corpus.sh && rm -rf corpus/linux/.git

COPY librarian/ librarian/
COPY data/ data/
COPY eval/ eval/
RUN python -m librarian.build_index

ENV PORT=8000 LIB_LOG_DIR=/app/logs
EXPOSE 8000
# 1 worker holds one copy of the index + models (~1 GB); threads serve concurrent users
CMD gunicorn librarian.wsgi:app --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 120
