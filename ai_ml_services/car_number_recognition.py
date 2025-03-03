import cv2
import numpy as np
import pytesseract

# Load the pre-trained Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

def preprocess_plate(plate_img):
    """ Preprocess the extracted plate image for better OCR accuracy. """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for contrast enhancement
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean small noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return morph

def extract_number_plate(image_path):
    """ Detects and extracts number plate text from an image. """
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect number plate using Haar Cascade
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))

    for (x, y, w, h) in plates:
        plate_img = image[y:y+h, x:x+w]

        # Preprocess the plate for OCR
        processed_plate = preprocess_plate(plate_img)

        # OCR with whitelist (Only uppercase letters and numbers)
        custom_config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        plate_text = pytesseract.image_to_string(processed_plate, config=custom_config)

        return plate_text.strip()  # Return the detected plate text

    return "No number plate detected"

# Example Usage:
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Change to your image path
    plate_number = extract_number_plate(image_path)
    print(f"Detected Number Plate: {plate_number}")
