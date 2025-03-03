from flask import Flask, render_template
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the uploads folder exists

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/car-number-recognition')
def car_number_recognition():
    return render_template('car_number_recognition.html')

if __name__ == '__main__':
    app.run(debug=True)
