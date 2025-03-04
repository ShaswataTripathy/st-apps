import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ai_ml_services.car_number_recognition import detect_and_recognize_plate

UPLOAD_FOLDER = "/app/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("car_number_recognition.html")

@app.route("/uploadImageForCarNumber", methods=["POST"])
def upload_image():
    """Handles image uploads and sends it for processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        try:
            file.save(filepath)
            if not os.path.exists(filepath):
                return jsonify({"error": "File not saved"})
            
            number_plate = detect_and_recognize_plate(filepath)
            return jsonify({"number_plate": number_plate})
        
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid file format. Allowed formats: png, jpg, jpeg"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
