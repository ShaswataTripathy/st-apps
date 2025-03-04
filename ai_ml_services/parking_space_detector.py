import cv2
import numpy as np
import base64

class ParkingSpaceDetector:
    def __init__(self, parking_lot_image):
        self.image = cv2.imread(parking_lot_image)
        self.original_image = self.image.copy()

    def detect_parking_spaces(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area to identify potential parking spaces
        parking_spaces = [cnt for cnt in contours if 500 < cv2.contourArea(cnt) < 5000]
        
        # Classify spaces as occupied or empty
        empty_spaces = 0
        for cnt in parking_spaces:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check if space is empty (simplified logic)
            roi = thresh[y:y+h, x:x+w]
            if np.mean(roi) > 200:  # If mostly white, consider it empty
                empty_spaces += 1
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for empty
            else:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for occupied
        
        return {
            'empty_spaces': empty_spaces,
            'total_spaces': len(parking_spaces),
            'marked_image': self._encode_image()
        }
    
    def _encode_image(self):
        # Encode image to base64 for web display
        _, buffer = cv2.imencode('.png', self.image)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
