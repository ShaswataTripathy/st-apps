import logging
import sys
import os
import glob
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ai_ml_services.car_number_recognition import (
    CarNumberRecognition, 
    validate_dependencies, 
    configure_logger
)

# Set matplotlib configuration directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp/matplotlib', exist_ok=True)

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

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


        # Clear other files in the uploads folder
        try:
            # Get all files in the uploads folder
            files_in_uploads = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            
            # Remove all files except the recently uploaded one
            for existing_file in files_in_uploads:
                if existing_file != filepath:
                    try:
                        os.remove(existing_file)
                        logger.info(f"Removed old file: {existing_file}")
                    except Exception as remove_error:
                        logger.warning(f"Could not remove file {existing_file}: {remove_error}")
        
        except Exception as clear_error:
            logger.error(f"Error clearing uploads folder: {clear_error}")
        
        # Process image
        try:
            recognizer = CarNumberRecognition()
            result = recognizer.process_image(filepath)
        except Exception as recognition_error:
            logger.error(f"Recognition error: {recognition_error}")
            result = {'error': str(recognition_error), 'total_plates': 0, 'plates': []}
    
        return jsonify(result)
    
    except Exception as e:
        # Log the error and return error response
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e), 'total_plates': 0, 'plates': []}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """
    Handle image analysis with advanced debugging
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.warning("No file uploaded for analysis")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logger.warning("No selected file for analysis")
            return jsonify({'error': 'No selected file'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform image analysis
            analysis_result = perform_image_analysis(filepath)
            
            # Remove uploaded file
            os.remove(filepath)
            
            return jsonify(analysis_result)
        
        except Exception as analysis_error:
            logger.error(f"Image analysis error: {analysis_error}")
            
            # Remove file if analysis fails
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({'error': str(analysis_error)}), 500
    
    except Exception as e:
        logger.error(f"Image analysis upload error: {e}")
        return jsonify({'error': str(e)}), 500

def perform_image_analysis(image_path):
    """
    Perform comprehensive image analysis with debugging information
    """
    import cv2
    import numpy as np
    import base64
    import matplotlib.pyplot as plt
    import io

    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a figure for visualization
    plt.figure(figsize=(15, 10))
    
    # Original Image
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Grayscale
    plt.subplot(2, 3, 2)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    
    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)
    plt.subplot(2, 3, 3)
    plt.title('Histogram Equalization')
    plt.imshow(hist_eq, cmap='gray')
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    plt.subplot(2, 3, 4)
    plt.title('Adaptive Thresholding')
    plt.imshow(adaptive_thresh, cmap='gray')
    
    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    plt.subplot(2, 3, 5)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    
    # Plate Detection Visualization
    recognizer = CarNumberRecognition()
    preprocessed_images = recognizer.preprocess_image(image)
    
    plate_detection_img = image.copy()
    for preprocessed_img in preprocessed_images:
        plates = recognizer.detect_plates(preprocessed_img)
        
        for (x, y, w, h) in plates:
            cv2.rectangle(plate_detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    plt.subplot(2, 3, 6)
    plt.title('Plate Detection')
    plt.imshow(cv2.cvtColor(plate_detection_img, cv2.COLOR_BGR2RGB))
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode plot to base64
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return {
        'width': image.shape[1],
        'height': image.shape[0],
        'mean_brightness': np.mean(gray),
        'std_deviation': np.std(gray),
        'debug_plot': f'data:image/png;base64,{plot_base64}'
    }
@app.errorhandler(Exception)
def handle_error(e):
    """
    Global error handler
    """
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': str(e), 'total_plates': 0, 'plates': []}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
