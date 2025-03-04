import cv2
import numpy as np
import pytesseract
import re
import logging
import sys
import imutils

class AdvancedCarNumberRecognition:
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
        
        # Noise reduction
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection preprocessing
        preprocessed_images = [
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
            
            # Canny edge detection
            cv2.Canny(gray, 170, 200)
        ]
        
        return preprocessed_images

    def detect_plates(self, image):
        """
        Multiple plate detection techniques
        """
        detected_plates = []
        
        # Technique 1: Haar Cascade
        try:
            plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            plates_haar = plate_cascade.detectMultiScale(
                image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(75, 25)
            )
            detected_plates.extend(plates_haar)
        except Exception as e:
            self.logger.warning(f"Haar Cascade detection failed: {e}")

        # Technique 2: Contour-based detection
        try:
            # Find contours
            contoured_image = image.copy()
            contours, _ = cv2.findContours(
                contoured_image, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours
            potential_plates = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Check if contour is roughly rectangular
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Typical license plate aspect ratios
                    if 2 < aspect_ratio < 6 and w > 50 and h > 20:
                        potential_plates.append((x, y, w, h))
            
            detected_plates.extend(potential_plates)
        except Exception as e:
            self.logger.warning(f"Contour-based detection failed: {e}")

        # Technique 3: Multiple image rotations
        try:
            rotations = [0, 15, -15, 30, -30]
            for angle in rotations:
                rotated = imutils.rotate_bound(image, angle)
                contours, _ = cv2.findContours(
                    rotated, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    if 2 < aspect_ratio < 6 and w > 50 and h > 20:
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
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 13'
        ]
        
        recognized_plates = []
        
        # Preprocessing techniques
        preprocessing_methods = [
            lambda x: x,  # Original image
            lambda x: cv2.equalizeHist(x),  # Histogram equalization
            lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
                    
                    # Basic length validation
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

# Standalone testing function
def test_plate_recognition(image_path):
    """
    Test plate recognition for a single image
    """
    recognizer = AdvancedCarNumberRecognition()
    result = recognizer.process_image(image_path)
    print("Plate Recognition Results:")
    print(result)
    return result

# If script is run directly
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test_plate_recognition(sys.argv[1])
    else:
        print("Please provide an image path")
