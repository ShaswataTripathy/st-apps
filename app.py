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

from ai_ml_services.car_number_recognition import extract_number_plate

@app.route("/uploadImageForCarNumber", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filepath = os.path.join("uploads", file.filename)  # Save to 'uploads' folder
    file.save(filepath)

    if not os.path.exists(filepath):
        return jsonify({"error": "File was not saved properly."})  # Additional check

    result = process_image(filepath)
        # Delete the file after processing
    try:
        os.remove(filepath)
    except Exception as e:
        print(f"Error deleting file: {e}")

    return jsonify(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
