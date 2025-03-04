import cv2
import numpy as np
import pytesseract
import re
import logging
import sys
import os

# Configure logging to use stdout and avoid file logging
def configure_logger():
    """
    Create a logger that writes to stdout and handles permission issues
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = configure_logger()

class CarNumberRecognition:
    def __init__(self):
        try:
            # Ensure Tesseract path is correctly set
            pytesseract.pytesseract.tesseract_cmd = self._find_tesseract_path()
            logger.info("Tesseract initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            raise

    def _find_tesseract_path(self):
        """
        Find Tesseract executable path
        """
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Tesseract executable not found")

    def process_image(self, image_path):
        """
        Simplified image processing pipeline
        """
        try:
            # Validate image path
            if not image_path or not isinstance(image_path, str):
                raise ValueError("Invalid image path")
            
            # Read the image
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            
            # Detect plates using Haar Cascade
            plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            
            plates = plate_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(75, 25)
            )
            
            # Recognize plates
            detected_plates = []
            for (x, y, w, h) in plates:
                try:
                    # Extract plate region
                    plate_img = original_image[y:y+h, x:x+w]
                    
                    # Perform OCR
                    plate_text = self._recognize_plate(plate_img)
                    
                    if plate_text:
                        detected_plates.append({
                            'plate': plate_text,
                            'location': [int(x), int(y), int(w), int(h)]
                        })
                
                except Exception as plate_e:
                    logger.warning(f"Individual plate processing failed: {plate_e}")
            
            # Log results
            logger.info(f"Processed image: {image_path}, Detected plates: {len(detected_plates)}")
            
            return {
                'plates': detected_plates,
                'total_plates': len(detected_plates)
            }
        
        except Exception as e:
            logger.error(f"Image processing completely failed: {e}")
            return {
                'error': str(e), 
                'total_plates': 0, 
                'plates': []
            }

    def _recognize_plate(self, plate_img):
        """
        Recognize plate text
        """
        try:
            # Convert plate image to grayscale
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh_plate = cv2.threshold(
                gray_plate, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Perform OCR
            ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(
                thresh_plate, 
                config=ocr_config
            ).strip()
            
            # Clean text
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Validate plate length
            return cleaned_text if 4 <= len(cleaned_text) <= 10 else None
        
        except Exception as e:
            logger.warning(f"Plate recognition failed: {e}")
            return None

def validate_dependencies():
    """
    Validate critical dependencies before application startup
    """
    try:
        # Check OpenCV
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
        
        # Check NumPy
        import numpy as np
        logger.info(f"NumPy Version: {np.__version__}")
        
        # Check Pytesseract
        import pytesseract
        logger.info(f"Pytesseract Version: {pytesseract.__version__}")
        
        # Verify Tesseract installation
        pytesseract.get_tesseract_version()
        
        return True
    
    except Exception as e:
        logger.error(f"Dependency validation failed: {e}")
        return False
