# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend and models
COPY backend ./backend
COPY models ./models

# Copy frontend build (assumes you've already built it locally)
COPY frontend/static/react-build ./frontend/static/react-build

# Copy any other necessary files
COPY backend/synthetic_smart_home_components.csv ./backend/
COPY .env ./

# Expose port
EXPOSE 5000

# Start the Flask app with Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "backend.app:app"]