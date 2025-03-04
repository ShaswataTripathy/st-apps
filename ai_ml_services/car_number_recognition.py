import os
import cv2
import torch
import easyocr
from ultralytics import YOLO

# Define model paths
EASYOCR_STORAGE_DIR = "/app/easyocr_model"
EASYOCR_USER_NETWORK_DIR = "/app/easyocr_user_network"

# Load YOLOv8 model for number plate detection
model_path = "ai_ml_services/yolov8n.pt"
try:
    yolo_model = YOLO(model_path)
    yolo_model.to("cpu")
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")

# Initialize EasyOCR reader
reader = easyocr.Reader(
    ['en'],
    model_storage_directory=EASYOCR_STORAGE_DIR,
    user_network_directory=EASYOCR_USER_NETWORK_DIR
)

def detect_and_recognize_plate(image_path):
    """Detects number plate using YOLO and recognizes characters using EasyOCR."""
    try:
        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Unable to read image"

        # Run YOLO model to detect license plates
        results = yolo_model(image)

        for result in results:
            boxes = result.boxes.xyxy  # Extract bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Get coordinates
                plate_crop = image[y1:y2, x1:x2]  # Crop the detected plate

                # Save the cropped plate for debugging (optional)
                cropped_plate_path = image_path.replace(".", "_plate.")
                cv2.imwrite(cropped_plate_path, plate_crop)

                # Run EasyOCR on the cropped image
                plate_text = reader.readtext(plate_crop)
                detected_texts = [text[1] for text in plate_text]

                if detected_texts:
                    return detected_texts[0]  # Return first detected text
        return "No plate detected"
    
    except Exception as e:
        return f"Error processing image: {str(e)}"
