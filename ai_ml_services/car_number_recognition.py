import cv2
import pytesseract
import numpy as np

# Ensure you have the Tesseract OCR path set correctly
# Uncomment the line below and set the path to where Tesseract is installed on your system
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(image_path):
    """Load an image from a specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the specified path: {image_path}")
    return image

def enhance_image(image):
    """Enhance the image for better number plate detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    hist_eq = cv2.equalizeHist(gray)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(hist_eq, (5, 5), 0)
    return blurred

def detect_plate(image):
    """Detect number plate in the image using Haar cascades."""
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')  # Make sure to have this XML file
    plates = plate_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    if len(plates) == 0:
        return None
    
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plate_region = image[y:y + h, x:x + w]
        return plate_region

def recognize_plate(plate_image):
    """Use Tesseract to recognize text on the number plate."""
    # Convert to grayscale for better OCR results
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use Tesseract to do OCR on the processed plate image
    config = '--psm 8'  # Assume single line of text
    number_plate_text = pytesseract.image_to_string(thresh_plate, config=config)
    
    return number_plate_text.strip()

def process_image(image_path):
    """Main function to run the number plate recognition."""
    image = load_image(image_path)
    enhanced_image = enhance_image(image)
    plate_image = detect_plate(enhanced_image)
    
    if plate_image is not None:
        number_plate_text = recognize_plate(plate_image)
        print(f"Detected Number Plate: {number_plate_text}")
    else:
        print("No number plate detected.")

if __name__ == "__main__":
    # Update this path with the path of the image you want to test
    image_path = "path_to_image.jpg"
    main(image_path)
