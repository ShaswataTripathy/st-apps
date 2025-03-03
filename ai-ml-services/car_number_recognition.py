import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    """ Load and preprocess the image for better OCR accuracy. """
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return thresh

def detect_number_plate(image):
    """ Detects the potential number plate region using contours. """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []
    for contour in contours:
        # Approximate contour and filter by shape
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Possible rectangular plate
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 6:  # Typical plate aspect ratio
                plate_candidates.append((x, y, w, h))

    return plate_candidates

def extract_text_from_plate(image, plate_regions):
    """ Extracts text from detected number plate regions. """
    detected_numbers = []
    for (x, y, w, h) in plate_regions:
        plate_roi = image[y:y+h, x:x+w]  # Crop the plate region

        # Further process the plate region
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Extract text using Tesseract
        text = pytesseract.image_to_string(plate_thresh, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        clean_text = ''.join(filter(str.isalnum, text))  # Keep only alphanumeric
        if clean_text:
            detected_numbers.append(clean_text)

    return detected_numbers

def process_image(image_path):
    """ Main function to process the image and extract number plates. """
    processed_image = preprocess_image(image_path)
    plate_regions = detect_number_plate(processed_image)

    if not plate_regions:
        return ["No plate detected"]

    # Reload original image for extracting text
    original_image = cv2.imread(image_path)
    plate_texts = extract_text_from_plate(original_image, plate_regions)

    return plate_texts if plate_texts else ["No text detected"]

