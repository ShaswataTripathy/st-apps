# Use an official Python runtime as a parent image
FROM python:3.9

# Set environment variables to avoid permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (fixes fontconfig issue)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (fixes package metadata issues)
RUN pip install --upgrade pip setuptools wheel

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Ultralytics model weights are available locally
RUN mkdir -p /app/ai_ml_services && \
    curl -L -o /app/ai_ml_services/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Expose port 7860 for Gradio / Flask applications
EXPOSE 7860

# Run the application
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "app:app"]
