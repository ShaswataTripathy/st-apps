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
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image):
        """
        Advanced image preprocessing techniques
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing techniques
        preprocessing_methods = [
            # Original grayscale
            gray,
            
            # Histogram equalization
            cv2.equalizeHist(gray),
            
            # Adaptive thresholding
            cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            ),
            
            # Noise reduction
            cv2.bilateralFilter(gray, 11, 17, 17),
            
            # Canny edge detection
            cv2.Canny(gray, 170, 200)
        ]
        
        return preprocessing_methods

    def detect_plates(self, image):
        """
        Multiple advanced plate detection techniques
        """
        detected_plates = []
        
        # Technique 1: Contour-based detection with multiple approaches
        try:
            # Find contours with different preprocessing
            preprocessed_images = [
                image,
                cv2.equalizeHist(image),
                cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ]
            
            for preprocessed in preprocessed_images:
                # Find contours
                contours, _ = cv2.findContours(
                    preprocessed, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Filter and validate contours
                for contour in contours:
                    # Compute contour properties
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    # Rectangular shape check
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Advanced plate aspect ratio validation
                        aspect_ratio = w / float(h)
                        area = w * h
                        
                        # Flexible plate detection criteria
                        if (2 < aspect_ratio < 6 and 
                            20 < w < 300 and 
                            20 < h < 100 and 
                            area > 500):
                            detected_plates.append((x, y, w, h))
        
        except Exception as e:
            self.logger.warning(f"Contour-based detection failed: {e}")
        
        # Technique 2: Multiple rotation and scaling
        try:
            rotations = [0, 15, -15, 30, -30]
            for angle in rotations:
                # Rotate image
                rotated = imutils.rotate_bound(image, angle)
                
                # Find contours in rotated image
                contours, _ = cv2.findContours(
                    rotated, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Plate-like region detection
                    if (2 < aspect_ratio < 6 and 
                        20 < w < 300 and 
                        20 < h < 100):
                        detected_plates.append((x, y, w, h))
        
        except Exception as e:
            self.logger.warning(f"Rotational detection failed: {e}")
        
        return list(set(detected_plates))

    def recognize_plate(self, plate_img):
        """
        Advanced plate recognition with multiple techniques
        """
        # Multiple OCR configurations
        ocr_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        recognized_plates = []
        
        # Preprocessing techniques
        preprocessing_methods = [
            lambda x: x,  # Original image
            lambda x: cv2.equalizeHist(x),  # Histogram equalization
            lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda x: cv2.GaussianBlur(x, (3, 3), 0)  # Gaussian blur
        ]
        
        for preprocess in preprocessing_methods:
            for config in ocr_configs:
                try:
                    # Preprocess plate image
                    processed_plate = preprocess(
                        cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) 
                        if len(plate_img.shape) == 3 
                        else plate_img
                    )
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        processed_plate, 
                        config=config
                    ).strip()
                    
                    # Clean and validate text
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    # Flexible length validation
                    if 4 <= len(cleaned_text) <= 10:
                        recognized_plates.append(cleaned_text)
                
                except Exception as e:
                    self.logger.warning(f"OCR attempt failed: {e}")
        
        # Return most frequent plate text
        return max(set(recognized_plates), key=recognized_plates.count) if recognized_plates else None

    def process_image(self, image_path):
        """
        Comprehensive image processing pipeline
        """
        try:
            # Read the image
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            
            # Preprocess images
            preprocessed_images = self.preprocess_image(original_image)
            
            # Store detected plates
            final_detected_plates = []
            
            # Process each preprocessed image
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
                            final_detected_plates.append({
                                'plate': plate_text,
                                'location': [int(x), int(y), int(w), int(h)]
                            })
                    
                    except Exception as plate_e:
                        self.logger.warning(f"Individual plate processing failed: {plate_e}")
            
            # Log and return results
            self.logger.info(f"Processed image: Total plates detected: {len(final_detected_plates)}")
            
            return {
                'plates': final_detected_plates,
                'total_plates': len(final_detected_plates)
            }
        
        except Exception as e:
            self.logger.error(f"Comprehensive image processing failed: {e}")
            return {
                'error': str(e), 
                'total_plates': 0, 
                'plates': []
            }


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
