import cv2
import numpy as np
import pytesseract
import os

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def detect_helmet(image):
    # Load a pre-trained Haar Cascade for upper body detection
    # You may need to replace this with a more specific helmet detection model
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in upper_bodies:
        # Check the top portion of the detected upper body for a helmet
        helmet_region = gray[y:y+int(h/2), x:x+w]
        if np.mean(helmet_region) < 100:  # Adjust this threshold as needed
            return True  # Helmet detected
    return False  # No helmet detected

def extract_number_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Adjust aspect ratio range for license plates
        if 2 < aspect_ratio < 5 and w > 100 and h > 20:
            license_plate = gray[y:y + h, x:x + w]
            
            # Apply additional preprocessing
            license_plate = cv2.GaussianBlur(license_plate, (3, 3), 0)
            license_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Use Tesseract with custom configuration
            text = pytesseract.image_to_string(license_plate, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if text and len(text.strip()) > 5:  # Ensure we have a substantial text detection
                return text.strip(), (x, y, w, h)
    
    return "", None

def process_frames(frame_folder, output_folder):
    if not os.path.exists(frame_folder):
        print(f"Frame folder does not exist: {frame_folder}")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for frame_file in frame_files:
        image = cv2.imread(frame_file)
        if image is None:
            print(f"Failed to load image: {frame_file}")
            continue
        
        # Check for helmet
        helmet_detected = detect_helmet(image)
        
        if not helmet_detected:
            # Extract number plate text only if no helmet is detected
            number_plate_text, plate_coords = extract_number_plate_text(image)
            
            if number_plate_text:
                # Draw rectangle around the number plate
                if plate_coords:
                    x, y, w, h = plate_coords
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Put text on the image
                cv2.putText(image, number_plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Extracted number plate text: {number_plate_text}")
            else:
                print(f"No number plate text detected in frame: {frame_file}")
        else:
            print(f"Helmet detected in frame: {frame_file}")
        
        # Add a red border around the entire image
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 10)
        
        output_path = os.path.join(output_folder, os.path.basename(frame_file))
        cv2.imwrite(output_path, image)
        print(f"Processed frame: {frame_file}")

# Replace with the path to the folder where the frames are stored
frame_folder = r"D:\\helmet\\Photos"
# Replace with the path to the folder where the output images will be saved
output_folder = r"D:\\helmet\\"

process_frames(frame_folder, output_folder)