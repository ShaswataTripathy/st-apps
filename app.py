import logging
import sys
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ai_ml_services.car_number_recognition import (
    CarNumberRecognition, 
    validate_dependencies, 
    configure_logger
)

logger = configure_logger()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists and has the right permissions
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Validate dependencies at startup
if not validate_dependencies():
    logger.critical("Dependency validation failed. Exiting.")
    sys.exit(1)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/car-number-recognition')
def car_number_recognition():
    return render_template('car_number_recognition.html')

@app.route('/uploadImageForCarNumber', methods=['POST'])
def upload_image():
    """
    Handle image upload and car number recognition
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.warning("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        try:
            recognizer = CarNumberRecognition()
            result = recognizer.process_image(filepath)
        except Exception as recognition_error:
            logger.error(f"Recognition error: {recognition_error}")
            result = {'error': str(recognition_error), 'total_plates': 0, 'plates': []}
        
        # Remove uploaded file after processing
        try:
            os.remove(filepath)
        except Exception as remove_error:
            logger.warning(f"Could not remove file: {remove_error}")
        
        return jsonify(result)
    
    except Exception as e:
        # Log the error and return error response
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e), 'total_plates': 0, 'plates': []}), 500

@app.errorhandler(Exception)
def handle_error(e):
    """
    Global error handler
    """
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': str(e), 'total_plates': 0, 'plates': []}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
