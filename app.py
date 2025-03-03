from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append(".")

from ai_ml_services.car_number_recognition import extract_number_plate


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
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Get detected plate
    plate_text = extract_number_plate(filepath)

    # Return data as a list
    return jsonify({"plates": [plate_text]})  # Always return a list


if __name__ == '__main__':
    app.run(debug=True)
