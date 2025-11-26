FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements & install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port untuk FastAPI
EXPOSE 8000

# Run API (FastAPI)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]