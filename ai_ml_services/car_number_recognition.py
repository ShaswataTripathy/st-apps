import cv2
import numpy as np
import pytesseract
import imutils

def preprocess_image(image_path):
    """Preprocess the image to enhance number plate detection."""
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to remove noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply edge detection
    edged = cv2.Canny(gray, 30, 200)
    
    return image, gray, edged

def extract_number_plate(image):
    """Extract number plate using contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to extract text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Sort contours by area and pick the largest 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    number_plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Looking for rectangular shape (number plate)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 6:  # Typical number plate aspect ratio
                number_plate = image[y:y+h, x:x+w]
                break

    return number_plate

def recognize_text(image):
    """Run OCR to extract text from the image."""
    if image is None:
        return "No plate detected"

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # OCR using Tesseract with a whitelist
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8'
    text = pytesseract.image_to_string(gray, config=custom_config)

    # Clean up text
    text = ''.join(filter(str.isalnum, text))
    return text if text else "No plate detected"

def process_image(image_path):
    """Process the uploaded image and return the detected number plate."""
    image, gray, edged = preprocess_image(image_path)
    number_plate_img = extract_number_plate(image)
    plate_text = recognize_text(number_plate_img)

    return {"plates": [plate_text]}
