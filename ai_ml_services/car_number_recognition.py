import cv2
import numpy as np
import pytesseract
import imutils
import re

def advanced_preprocess_image(image):
    """
    Enhanced image preprocessing with multiple techniques
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques
    preprocessed_images = [
        # Original grayscale
        gray,
        
        # Histogram Equalization
        cv2.equalizeHist(gray),
        
        # Adaptive Histogram Equalization
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
        
        # Gaussian Blur
        cv2.GaussianBlur(gray, (5, 5), 0),
        
        # Median Blur
        cv2.medianBlur(gray, 3)
    ]
    
    return preprocessed_images

def detect_plates_advanced(image):
    """
    Advanced plate detection with multiple techniques
    """
    # Multiple plate detection methods
    plate_detection_methods = [
        # Haar Cascade Classifier
        lambda img: cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        ).detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(75, 25)),
        
        # Contour-based detection
        lambda img: find_plate_contours(img)
    ]
    
    detected_plates = []
    
    # Try each detection method
    for method in plate_detection_methods:
        try:
            plates = method(image)
            if len(plates) > 0:
                detected_plates.extend(plates)
        except Exception as e:
            print(f"Detection method failed: {e}")
    
    return detected_plates

def find_plate_contours(image):
    """
    Contour-based plate detection
    """
    # Apply edge detection
    edges = cv2.Canny(image, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
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

def enhance_plate_image(plate_img):
    """
    Advanced plate image enhancement
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Multiple enhancement techniques
    enhancement_methods = [
        # Original grayscale
        gray,
        
        # Adaptive thresholding
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2),
        
        # Otsu's thresholding
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        
        # Sharpening
        cv2.filter2D(gray, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    ]
    
    return enhancement_methods

def recognize_plate_advanced(plate_images):
    """
    Advanced plate recognition with multiple OCR configurations
    """
    recognized_plates = []
    
    # Multiple Tesseract configurations
    ocr_configs = [
        r'--oem 3 --psm 6',  # Assume a single uniform block of text
        r'--oem 3 --psm 8',  # Treat the image as a single word
        r'--oem 3 --psm 11',  # Sparse text with OSD
        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    ]
    
    for plate_img in plate_images:
        plate_texts = []
        
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(plate_img, config=config).strip()
                
                # Clean and validate text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if 2 < len(cleaned_text) < 10:  # Basic length validation
                    plate_texts.append(cleaned_text)
            except Exception as e:
                print(f"OCR failed with config {config}: {e}")
        
        # Take the most frequent or first valid plate text
        if plate_texts:
            recognized_plate = max(set(plate_texts), key=plate_texts.count)
            recognized_plates.append(recognized_plate)
    
    return recognized_plates

def process_image(image_input):
    """
    Comprehensive image processing for plate recognition
    """
    try:
        # Handle different input types
        if isinstance(image_input, str):
            original_image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            original_image = image_input
        else:
            # Assume file-like object
            import numpy as np
            image_bytes = image_input.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return {"error": "Unable to read image"}
        
        # Preprocess images
        preprocessed_images = advanced_preprocess_image(original_image)
        
        # Detected plates across different preprocessed images
        all_detected_plates = []
        
        # Try detection on each preprocessed image
        for preprocessed_img in preprocessed_images:
            plates = detect_plates_advanced(preprocessed_img)
            
            for (x, y, w, h) in plates:
                # Extract plate region
                plate_img = original_image[y:y+h, x:x+w]
                
                # Enhance plate images
                enhanced_plates = enhance_plate_image(plate_img)
                
                # Recognize plates
                recognized_plates = recognize_plate_advanced(enhanced_plates)
                
                for plate_text in recognized_plates:
                    all_detected_plates.append({
                        "plate": plate_text,
                        "location": (x, y, w, h)
                    })
        
        # Remove duplicate plates
        unique_plates = list({plate['plate']: plate for plate in all_detected_plates}.values())
        
        return {
            "plates": unique_plates,
            "total_plates": len(unique_plates)
        }
    
    except Exception as e:
        return {"error": str(e)}

# Debugging function
def debug_plate_detection(image_path):
    """
    Detailed debugging for plate detection
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Preprocess images
    preprocessed_images = advanced_preprocess_image(image)
    
    # Create a figure to show different preprocessing stages
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    titles = [
        'Original Grayscale', 
        'Histogram Equalization', 
        'Adaptive Histogram Equalization', 
        'Gaussian Blur', 
        'Median Blur'
    ]
    
    for i, (img, title) in enumerate(zip(preprocessed_images, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        
        # Detect plates on each preprocessed image
        try:
            plates = detect_plates_advanced(img)
            plt.title(f'{title}\nPlates Detected: {len(plates)}')
        except Exception as e:
            plt.title(f'{title}\nError: {e}')
    
    plt.tight_layout()
    plt.show()

