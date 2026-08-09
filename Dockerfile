FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and schemas
COPY schema/ schema/
COPY takehome_service/ takehome_service/
COPY app.py .

# Environment defaults — overridable by compose / container runner
ENV BOOK_PATH=/data/client_book.json \
    MARKET_PATH=/data/market_data.json \
    LLM_BASE_URL=http://localhost:8600/v1 \
    LLM_API_KEY=test \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
