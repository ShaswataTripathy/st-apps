import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO

# Load YOLOv8 model for number plate detection
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

# Load the model from local file
model = YOLO(MODEL_PATH) # Using the nano version for efficiency
reader = easyocr.Reader(['en'])  # EasyOCR for text recognition

def detect_plate(image):
    """ Detects number plate using YOLOv8 """
    results = model(image)
    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2))
    return plates

def extract_text(image, plates):
    """ Extracts text from the detected license plates """
    detected_numbers = []
    for (x1, y1, x2, y2) in plates:
        plate_roi = image[y1:y2, x1:x2]
        text_results = reader.readtext(plate_roi, detail=0)
        detected_numbers.extend(text_results)
    return detected_numbers if detected_numbers else ["No plate text detected"]

def process_image(image_path):
    """ Full pipeline: detect plate → extract text """
    image = cv2.imread(image_path)
    plates = detect_plate(image)
    
    if plates:
        return extract_text(image, plates)
    return ["No plate detected"]
