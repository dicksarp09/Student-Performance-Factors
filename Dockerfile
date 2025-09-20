# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install DVC (with support for all storage backends, optional)
RUN pip install --no-cache-dir dvc[all]

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 120

# Copy project files
COPY . .

# Copy models explicitly into /app/models
COPY models/ /app/models/

# Expose ports for Flask and MLflow
EXPOSE 5002
EXPOSE 5000

CMD ["python", "api/dvc_mlflow_app.py"]
