from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append(".")

from ai_ml_services.car_number_recognition import process_image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the uploads folder exists

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/car-number-recognition')
def car_number_recognition():
    return render_template('car_number_recognition.html')

@app.route('/uploadImageForCarNumber', methods=['POST'])
def upload_image_for_car_number():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    plates = process_image(filepath)
    return jsonify({'plates': plates})

if __name__ == '__main__':
    app.run(debug=True)
