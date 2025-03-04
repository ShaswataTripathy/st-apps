import cv2
import numpy as np
import pytesseract
import re
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('car_number_recognition.log')
    ]
)
logger = logging.getLogger(__name__)

class CarNumberRecognition:
    def __init__(self):
        try:
            # Ensure Tesseract path is correctly set
            pytesseract.pytesseract.tesseract_cmd = self.find_tesseract_path()
            logger.info("Tesseract initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            raise

    def find_tesseract_path(self):
        """
        Find Tesseract executable path
        """
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        
        for path in possible_paths:
            try:
                # Verify tesseract version
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract found at {path}. Version: {version}")
                return path
            except Exception:
                continue
        
        raise FileNotFoundError("Tesseract executable not found")

    def preprocess_image(self, image):
        """
        Advanced image preprocessing with error handling
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Preprocessing techniques
            preprocessed_images = [
                gray,
                cv2.equalizeHist(gray),
                cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2),
                cv2.GaussianBlur(gray, (5, 5), 0)
            ]
            
            return preprocessed_images
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return [image]  # Fallback to original image

    def detect_plates(self, image):
        """
        Detect license plates with multiple fallback methods
        """
        try:
            # Primary: Haar Cascade Classifier
            plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            
            plates = plate_cascade.detectMultiScale(
                image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(75, 25)
            )
            
            # Fallback: Contour-based detection if Haar fails
            if len(plates) == 0:
                plates = self._contour_plate_detection(image)
            
            return plates
        
        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            return []

    def _contour_plate_detection(self, image):
        """
        Fallback plate detection using contours
        """
        try:
            # Edge detection
            edges = cv2.Canny(image, 100, 200)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter potential plate contours
            plate_contours = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Check if contour is roughly rectangular
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Typical license plate aspect ratios
                    if 2 < aspect_ratio < 5 and w > 50 and h > 20:
                        plate_contours.append((x, y, w, h))
            
            return plate_contours
        
        except Exception as e:
            logger.error(f"Contour-based detection failed: {e}")
            return []

    def recognize_plate(self, plate_img):
        """
        Advanced plate recognition with multiple OCR techniques
        """
        # Multiple OCR configurations
        ocr_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 11'
        ]
        
        recognized_plates = []
        
        for config in ocr_configs:
            try:
                # Perform OCR
                text = pytesseract.image_to_string(plate_img, config=config).strip()
                
                # Clean and validate text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # Basic length validation
                if 4 <= len(cleaned_text) <= 10:
                    recognized_plates.append(cleaned_text)
            
            except Exception as e:
                logger.warning(f"OCR failed with config {config}: {e}")
        
        # Return most frequent plate text
        return max(set(recognized_plates), key=recognized_plates.count) if recognized_plates else None

    def process_image(self, image_path):
        """
        Comprehensive image processing pipeline with robust error handling
        """
        try:
            # Validate image path
            if not image_path or not isinstance(image_path, str):
                raise ValueError("Invalid image path")
            
            # Read the image
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            
            # Preprocess images
            preprocessed_images = self.preprocess_image(original_image)
            
            # Store detected plates
            detected_plates = []
            
            # Try detection on each preprocessed image
            for preprocessed_img in preprocessed_images:
                # Detect plates
                plates = self.detect_plates(preprocessed_img)
                
                # Process each detected plate
                for (x, y, w, h) in plates:
                    try:
                        # Extract plate region
                        plate_img = original_image[y:y+h, x:x+w]
                        
                        # Recognize plate
                        plate_text = self.recognize_plate(plate_img)
                        
                        # Store if plate text is found
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

# Validation function to check dependencies
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
        logger.critical(f"Dependency validation failed: {e}")
        return False

# Run dependency check
if not validate_dependencies():
    logger.critical("Critical dependencies not met. Exiting.")
    sys.exit(1)
