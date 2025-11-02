# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI default port
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
