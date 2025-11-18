FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from build context
COPY requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8001

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
