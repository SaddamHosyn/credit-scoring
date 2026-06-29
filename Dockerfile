FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first to leverage Docker build cache layers
COPY requirements.txt requirements_mlops.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements_mlops.txt

# Copy source code and app folders
COPY app/ ./app/
COPY src/ ./src/

# Expose target port
EXPOSE 8000

# Run FastAPI app under production-ready server (Uvicorn with worker parameters)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
