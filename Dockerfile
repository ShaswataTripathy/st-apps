# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng  # Install English language model for Tesseract

# Copy all files to the container
COPY . /app

RUN mkdir -p /app/uploads && chown -R 1000:1000 /app/uploads

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Expose the port Flask will run on
EXPOSE 7860

# Start the Flask app using Gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "4", \
     "--timeout", "120", \
     "--log-level", "info", \
     "app:app"]
