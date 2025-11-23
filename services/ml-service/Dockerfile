FROM python:3.11-slim

WORKDIR /app

# Copy requirements from build context
COPY requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8002

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
