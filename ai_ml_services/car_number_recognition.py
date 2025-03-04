import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    """ Load and preprocess the image to enhance text detection. """
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise using Bilateral Filtering (better than GaussianBlur for edges)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return image, thresh

def detect_number_plate(original, processed):
    """ Detect potential license plate regions using contour detection. """
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Filter based on rectangular shape and aspect ratio
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 6:  # License plates usually have an aspect ratio within this range
                plate_candidates.append((x, y, w, h))

    # Return the best candidate if found
    return plate_candidates if plate_candidates else None

def extract_text_from_plate(image, plate_regions):
    """ Extract text from detected number plate regions using Tesseract OCR. """
    detected_numbers = []
    
    for (x, y, w, h) in plate_regions:
        plate_roi = image[y:y+h, x:x+w]  # Crop the plate region

        # Convert to grayscale
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoising
        plate_denoised = cv2.fastNlMeansDenoising(plate_thresh, None, 30, 7, 21)

        # OCR with optimized settings for license plates
        text = pytesseract.image_to_string(plate_denoised, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # Clean extracted text
        clean_text = ''.join(filter(str.isalnum, text))
        if clean_text:
            detected_numbers.append(clean_text)

    return detected_numbers if detected_numbers else ["No plate text detected"]

def process_image(image_path):
    """ Full pipeline to detect and recognize license plate numbers. """
    original_image, processed_image = preprocess_image(image_path)
    plate_regions = detect_number_plate(original_image, processed_image)

    if not plate_regions:
        return ["No plate detected"]

    # Extract text from detected plates
    plate_texts = extract_text_from_plate(original_image, plate_regions)

    return plate_texts
