import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
import json

def preprocess_image(image):
    """
    Advanced image preprocessing for better plate detection
    :param image: Input image (numpy array or PIL Image)
    :return: Preprocessed grayscale image
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)
    
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    
    return morphed, image

def detect_plates(preprocessed_image):
    """
    Detect number plates in the image
    :param preprocessed_image: Preprocessed grayscale image
    :return: List of detected plate regions
    """
    # Load Haar Cascade classifier
    plate_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    )
    
    # Detect plates using Haar Cascade
    plates = plate_cascade.detectMultiScale(
        preprocessed_image, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(75, 25)
    )
    
    return plates

def enhance_plate_image(plate_img):
    """
    Enhance the detected plate image for better OCR
    :param plate_img: Detected plate image
    :return: Enhanced plate image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Sharpen the image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened

def recognize_plate(plate_img):
    """
    Recognize text from the plate image
    :param plate_img: Enhanced plate image
    :return: Recognized plate number
    """
    # Tesseract configuration for better recognition
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Perform OCR
    plate_text = pytesseract.image_to_string(
        plate_img, 
        config=custom_config
    ).strip()
    
    # Clean and validate the plate number
    plate_text = validate_plate_number(plate_text)
    
    return plate_text

def validate_plate_number(plate_text):
    """
    Validate and clean the detected plate number
    :param plate_text: Raw detected plate text
    :return: Cleaned plate number
    """
    # Remove non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Additional validation rules
    # Example: Ensure plate is between 6-8 characters
    if 6 <= len(cleaned_text) <= 8:
        return cleaned_text
    
    return ""

def process_image(image_input):
    """
    Main processing method for number plate recognition
    :param image_input: Image file, file path, or PIL Image
    :return: Dictionary with recognition results
    """
    try:
        # Handle different input types
        if isinstance(image_input, str):
            # If it's a file path
            original_image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            # If it's a PIL Image
            original_image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            # If it's already a numpy array
            original_image = image_input
        elif hasattr(image_input, 'read'):
            # If it's a file-like object (e.g., from file upload)
            image_bytes = image_input.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            return {"error": "Unsupported image input type"}
        
        if original_image is None:
            return {"error": "Unable to read image"}
        
        # Preprocess the image
        preprocessed, original = preprocess_image(original_image)
        
        # Detect plates
        plates = detect_plates(preprocessed)
        
        # Store recognized plates
        recognized_plates = []
        
        # Process each detected plate
        for (x, y, w, h) in plates:
            # Extract plate region
            plate_img = original[y:y+h, x:x+w]
            
            # Enhance plate image
            enhanced_plate = enhance_plate_image(plate_img)
            
            # Recognize plate number
            plate_number = recognize_plate(enhanced_plate)
            
            if plate_number:
                recognized_plates.append({
                    "plate": plate_number,
                    "location": (x, y, w, h)
                })
        print ("len of recognized_plates " + json.dumps(recognized_plates) )
        return {
            "plates": recognized_plates,
            "total_plates": len(recognized_plates)
        }
    
    except Exception as e:
        return {"error": str(e)}

# Compatibility with various frameworks
def process_uploaded_image(uploaded_file):
    """
    Specific method for handling file uploads in web frameworks
    :param uploaded_file: Uploaded file object
    :return: Recognition results
    """
    return process_image(uploaded_file)
