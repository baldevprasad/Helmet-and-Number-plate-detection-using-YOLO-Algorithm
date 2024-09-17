import torch
import cv2
import numpy as np
import pytesseract
import os

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_bike_rider_and_helmet(image, model):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    
    bike_rider_detected = False
    helmet_detected = False
    bike_rider_box = None
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = model.names[int(class_id)]
        
        if label == 'person' and confidence > 0.3:
            bike_rider_detected = True
            bike_rider_box = (int(x1), int(y1), int(x2), int(y2))
            break
    
    if bike_rider_detected:
        head_top = bike_rider_box[1]
        head_bottom = bike_rider_box[1] + int((bike_rider_box[3] - bike_rider_box[1]) * 0.3)
        head_left = bike_rider_box[0]
        head_right = bike_rider_box[2]
        
        head_region = image[head_top:head_bottom, head_left:head_right]
        
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) > 8:
                area = cv2.contourArea(contour)
                if area > 500:  # Adjust this threshold based on your image size
                    helmet_detected = True
                    break
    
    return bike_rider_detected, helmet_detected, bike_rider_box

def extract_number_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        if 2 < aspect_ratio < 5 and w > 100 and h > 20:
            license_plate = gray[y:y + h, x:x + w]
            license_plate = cv2.GaussianBlur(license_plate, (3, 3), 0)
            license_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(license_plate, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if text and len(text.strip()) > 5:
                return text.strip(), (x, y, w, h)
    
    return "", None

def process_frames(frame_folder, output_folder):
    if not os.path.exists(frame_folder):
        print(f"Frame folder does not exist: {frame_folder}")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model = load_yolo_model()
    
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for frame_file in frame_files:
        image = cv2.imread(frame_file)
        if image is None:
            print(f"Failed to load image: {frame_file}")
            continue
        
        bike_rider_detected, helmet_detected, bike_rider_box = detect_bike_rider_and_helmet(image, model)
        
        if bike_rider_detected:
            color = (0, 255, 0) if helmet_detected else (0, 0, 255)
            cv2.rectangle(image, (bike_rider_box[0], bike_rider_box[1]), (bike_rider_box[2], bike_rider_box[3]), color, 2)
            text = "Helmet Detected" if helmet_detected else "No Helmet Detected"
            cv2.putText(image, text, (bike_rider_box[0], bike_rider_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            if not helmet_detected:
                number_plate_text, plate_coords = extract_number_plate_text(image)
                if number_plate_text:
                    if plate_coords:
                        x, y, w, h = plate_coords
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(image, number_plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    print(f"Extracted number plate text: {number_plate_text}")
                else:
                    print(f"No number plate text detected in frame: {frame_file}")
        else:
            cv2.putText(image, "No Bike Rider Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 10)
        
        output_path = os.path.join(output_folder, os.path.basename(frame_file))
        cv2.imwrite(output_path, image)
        print(f"Processed frame: {frame_file}")

# Replace with the path to the folder where the frames are stored
frame_folder = r"D:\helmet\Photos"
# Replace with the path to the folder where the output images will be saved
output_folder = r"D:\helmet\Output"

process_frames(frame_folder, output_folder)