from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append(".")

from ai_ml_services.car_number_recognition import process_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists and has the right permissions
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.chmod(app.config['UPLOAD_FOLDER'], 0o777)  # Set full read/write/execute permissions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/car-number-recognition')
def car_number_recognition():
    return render_template('car_number_recognition.html')

@app.route("/uploadImageForCarNumber", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    # Secure the filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        if not os.path.exists(filepath):
            return jsonify({"error": "File was not saved properly."})  # Additional check

        # Process the image
        result = process_image(filepath)

        # Delete the file after processing
        os.remove(filepath)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
