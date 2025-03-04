import cv2
import numpy as np
import pytesseract
import re
import logging
from functools import wraps
import signal
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout decorator
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                logger.warning(f"Function {func.__name__} timed out")
                return None
        return wrapper
    return decorator

class NumberPlateRecognition:
    def __init__(self, timeout_seconds=10):
        self.timeout_seconds = timeout_seconds
        # Ensure Tesseract path is correctly set
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    @timeout(10)
    def preprocess_image(self, image):
        """
        Efficient image preprocessing
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)
            
            return denoised
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None

    @timeout(10)
    def detect_plates(self, image):
        """
        Advanced plate detection with multiple techniques
        """
        try:
            # Haar Cascade Classifier
            plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            
            plates = plate_cascade.detectMultiScale(
                image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(75, 25)
            )
            
            return plates
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            return []

    @timeout(10)
    def enhance_plate_image(self, plate_img):
        """
        Plate image enhancement
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
        except Exception as e:
            logger.error(f"Plate enhancement error: {e}")
            return None

    @timeout(10)
    def recognize_plate(self, plate_img):
        """
        Advanced plate recognition
        """
        try:
            # Multiple OCR configurations
            ocr_configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            recognized_plates = []
            
            for config in ocr_configs:
                try:
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        plate_img, 
                        config=config
                    ).strip()
                    
                    # Clean text
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    # Validate plate length
                    if 4 <= len(cleaned_text) <= 10:
                        recognized_plates.append(cleaned_text)
                except Exception as inner_e:
                    logger.warning(f"OCR config error: {inner_e}")
            
            # Return most frequent plate or first valid
            return max(set(recognized_plates), key=recognized_plates.count) if recognized_plates else None
        
        except Exception as e:
            logger.error(f"Plate recognition error: {e}")
            return None

    def process_image(self, image_input):
        """
        Comprehensive image processing
        """
        start_time = time.time()
        
        try:
            # Read image
            if isinstance(image_input, str):
                original_image = cv2.imread(image_input)
            elif hasattr(image_input, 'read'):
                # File-like object
                import numpy as np
                image_bytes = image_input.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                original_image = image_input
            
            if original_image is None:
                return {"error": "Unable to read image", "total_plates": 0, "plates": []}
            
            # Preprocess
            preprocessed = self.preprocess_image(original_image)
            if preprocessed is None:
                return {"error": "Preprocessing failed", "total_plates": 0, "plates": []}
            
            # Detect plates
            plates = self.detect_plates(preprocessed)
            
            # Recognize plates
            detected_plates = []
            for (x, y, w, h) in plates:
                try:
                    # Extract plate region
                    plate_img = original_image[y:y+h, x:x+w]
                    
                    # Enhance plate
                    enhanced_plate = self.enhance_plate_image(plate_img)
                    if enhanced_plate is None:
                        continue
                    
                    # Recognize plate
                    plate_text = self.recognize_plate(enhanced_plate)
                    
                    if plate_text:
                        detected_plates.append({
                            "plate": plate_text,
                            "location": [int(x), int(y), int(w), int(h)]
                        })
                except Exception as plate_e:
                    logger.warning(f"Plate processing error: {plate_e}")
            
            # Remove duplicates
            unique_plates = list({plate['plate']: plate for plate in detected_plates}.values())
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Image processed in {processing_time:.2f} seconds")
            
            return {
                "plates": unique_plates,
                "total_plates": len(unique_plates)
            }
        
        except Exception as e:
            logger.error(f"Complete processing error: {e}")
            return {"error": str(e), "total_plates": 0, "plates": []}
            
