import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

def detailed_image_analysis(image_path):
    """
    Comprehensive image analysis and debugging
    """
    # Read the image
    original_image = cv2.imread(image_path)
    
    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # Original Image
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_rgb)
    
    # Grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    plt.subplot(3, 3, 2)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    
    # Histogram Equalization
    eq_gray = cv2.equalizeHist(gray)
    plt.subplot(3, 3, 3)
    plt.title('Histogram Equalized')
    plt.imshow(eq_gray, cmap='gray')
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    plt.subplot(3, 3, 4)
    plt.title('Adaptive Thresholding')
    plt.imshow(adaptive_thresh, cmap='gray')
    
    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    plt.subplot(3, 3, 5)
    plt.title('Canny Edge Detection')
    plt.imshow(edges, cmap='gray')
    
    # Plate Detection Attempt
    plate_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    )
    
    # Try multiple detection parameters
    detection_attempts = [
        (1.1, 3, (30, 10)),
        (1.3, 4, (50, 20)),
        (1.05, 5, (20, 5))
    ]
    
    detected_plates_image = original_rgb.copy()
    all_plates = []
    
    for scaleFactor, minNeighbors, minSize in detection_attempts:
        plates = plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=scaleFactor, 
            minNeighbors=minNeighbors, 
            minSize=minSize
        )
        all_plates.extend(plates)
        
        # Draw rectangles on detected plates
        for (x, y, w, h) in plates:
            cv2.rectangle(
                detected_plates_image, 
                (x, y), 
                (x+w, y+h), 
                (0, 255, 0), 
                2
            )
    
    plt.subplot(3, 3, 6)
    plt.title(f'Plate Detection (Total: {len(all_plates)})')
    plt.imshow(detected_plates_image)
    
    # Detailed Plate Extraction and OCR
    plt.subplot(3, 3, 7)
    plate_texts = []
    
    for (x, y, w, h) in all_plates:
        plate_img = gray[y:y+h, x:x+w]
        
        # Multiple OCR attempts
        ocr_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 11'
        ]
        
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(
                    plate_img, 
                    config=config
                ).strip()
                
                if text and len(text) >= 4:
                    plate_texts.append(text)
            except Exception as e:
                print(f"OCR attempt failed: {e}")
    
    plt.title('Plate OCR Results')
    plt.text(0.5, 0.5, '\n'.join(plate_texts) if plate_texts else 'No plates detected', 
             horizontalalignment='center', 
             verticalalignment='center')
    
    # Image Properties
    plt.subplot(3, 3, 8)
    plt.title('Image Properties')
    properties = [
        f'Shape: {original_image.shape}',
        f'Dtype: {original_image.dtype}',
        f'Mean Brightness: {np.mean(gray)}',
        f'Std Deviation: {np.std(gray)}'
    ]
    plt.text(0.5, 0.5, '\n'.join(properties), 
             horizontalalignment='center', 
             verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print detected plate texts
    print("\nDetected Plate Texts:")
    for text in plate_texts:
        print(text)
    
    return {
        'plates': plate_texts,
        'total_plates': len(plate_texts)
    }

# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detailed_image_analysis(sys.argv[1])
    else:
        print("Please provide an image path")
