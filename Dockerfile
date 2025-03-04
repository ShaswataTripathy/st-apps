# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to avoid permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV EASYOCR_STORAGE_DIR=/app/easyocr_model
ENV EASYOCR_USER_NETWORK_DIR=/app/easyocr_user_network

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Create a new user with UID 1000
RUN useradd -u 1000 appuser

# Create necessary directories with correct permissions
RUN mkdir -p /app/uploads /app/easyocr_model /app/easyocr_user_network /app/ai_ml_services \
    && chown -R appuser:appuser /app \
    && chmod -R 777 /app/uploads /app/easyocr_model /app/easyocr_user_network

# Switch to the new user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . /app

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Ensure Ultralytics model weights are available locally
COPY ai_ml_services/yolov8n.pt /app/ai_ml_services/yolov8n.pt

# Expose the necessary port
EXPOSE 7860

# Command to run the application
CMD ["python3", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "app:app"]
