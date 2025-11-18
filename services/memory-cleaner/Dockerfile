FROM python:3.11-slim

WORKDIR /app

# Copy requirements from build context
COPY requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "cleaner.py"]
