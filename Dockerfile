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
RUN mkdir -p /app/uploads /app/easyocr_model /app/easyocr_user_network \
    && chown -R appuser:appuser /app \
    && chmod -R 777 /app/uploads /app/easyocr_model /app/easyocr_user_network

# Install application dependencies before switching user
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Grant write permissions to appuser home directory
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser && chmod -R 777 /home/appuser

# Switch to the new user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . /app

# Expose the necessary port
EXPOSE 7860

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
