from flask import Flask, render_template, request, jsonify
import os
from ai_ml_services.car_number_recognition import process_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/car_number_recognition")
def car_number_recognition():
    return render_template("car_number_recognition.html")


@app.route("/uploadImageForCarNumber", methods=["POST"])
def upload_image_for_car_number():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    try:
        detected_number = process_image(filename)
        return jsonify({"number_plate": detected_number})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
