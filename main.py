# detector.py
#CODE1
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from PIL import Image
import multiprocessing as mp
import queue
import threading
import time
import torch
import os

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.25):
        """Initialize YOLO model and OCR reader"""
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Configure reader with specific parameters for license plates
        self.reader = easyocr.Reader(['en'], gpu=False, 
                                   model_storage_directory='.',
                                   recog_network='english_g2')
        
    def preprocess_plate(self, plate_img):
        """Enhanced preprocessing for license plate images"""
        if plate_img is None or plate_img.size == 0:
            return None
            
        try:
            # Resize with maintained aspect ratio
            height = 100
            aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
            width = int(height * aspect_ratio)
            plate_img = cv2.resize(plate_img, (width, height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Advanced image enhancement
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if np.mean(enhanced) > 127 else cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            print(f"Error in preprocessing plate: {e}")
            return None

    def find_nearest_plate(self, no_helmet_bbox, plate_bboxes, max_distance=150):
        """Find the nearest license plate to a rider without helmet"""
        if not plate_bboxes:
            return None
            
        nh_center = [(no_helmet_bbox[0] + no_helmet_bbox[2])/2,
                     (no_helmet_bbox[1] + no_helmet_bbox[3])/2]
        
        nearest_plate = None
        min_distance = float('inf')
        
        for plate_bbox in plate_bboxes:
            plate_center = [(plate_bbox[0] + plate_bbox[2])/2,
                          (plate_bbox[1] + plate_bbox[3])/2]
            
            # Calculate Euclidean distance
            distance = np.sqrt((nh_center[0] - plate_center[0])**2 +
                             (nh_center[1] - plate_center[1])**2)
            
            # Check if plate is below the rider (assuming camera perspective)
            if plate_center[1] > nh_center[1] and distance < min_distance:
                min_distance = distance
                nearest_plate = plate_bbox
        
        return nearest_plate if min_distance <= max_distance else None

    def read_license_plate(self, frame, bbox):
        """Improved license plate reading with better format handling"""
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Extract with minimal padding
            padding = 5
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            plate_img = frame[y1:y2, x1:x2]
            
            # Simple preprocessing - just grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Run OCR with adjusted parameters
            results = self.reader.readtext(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',  # Allow spaces
                batch_size=1,
                detail=0,
                paragraph=False,  # Don't combine as single text
                width_ths=1.0,    # Default width threshold
                mag_ratio=1.0     # No magnification
            )
            
            if results:
                # Combine all detected text
                text = ' '.join(results)
                
                # Clean the text
                # Remove any characters that aren't alphanumeric or spaces
                text = ''.join(c for c in text if c.isalnum() or c.isspace())
                
                # Split into parts and clean each part
                parts = text.split()
                cleaned_parts = []
                
                for i, part in enumerate(parts):
                    if i == 0:  # State code
                        cleaned_parts.append(part[:2].upper())  # First 2 chars for state
                    elif i == 1:  # District number
                        cleaned_parts.append(part[:2])  # 2 digits for district
                    elif i == 2:  # Letters
                        cleaned_parts.append(part[:2].upper())  # 2 letters
                    elif i == 3:  # Registration number
                        cleaned_parts.append(part[:4])  # Up to 4 digits
                
                # Format the final plate number with spaces
                if len(cleaned_parts) >= 4:
                    return ' '.join(cleaned_parts[:4])
                
                return text
                
        except Exception as e:
            print(f"Plate reading error: {e}")
            return None
            
        return None

    def is_plate_near_rider(self, rider_bbox, plate_bbox):
        """Check if a license plate is near a rider without helmet"""
        rx1, ry1, rx2, ry2 = map(float, rider_bbox)
        px1, py1, px2, py2 = map(float, plate_bbox)
        
        # Calculate centers
        rider_center_x = (rx1 + rx2) / 2
        rider_center_y = (ry1 + ry2) / 2
        plate_center_x = (px1 + px2) / 2
        plate_center_y = (py1 + py2) / 2
        
        # Calculate distances
        horizontal_distance = abs(rider_center_x - plate_center_x)
        vertical_distance = abs(rider_center_y - plate_center_y)
        
        # Define thresholds based on rider size
        rider_width = rx2 - rx1
        rider_height = ry2 - ry1
        
        # Check if plate is within reasonable distance
        return (horizontal_distance < rider_width * 1.5 and 
                vertical_distance < rider_height * 1.5)

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with detections"""
        violations = []
        license_plates = []
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold)[0]
        
        # Track detections by type
        bikes = []
        helmets = []  # Added helmet tracking
        no_helmets = []
        mirrors = []
        plate_boxes = []
        
        # First pass - collect all detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            class_name = results.names[int(class_id)]
            
            if class_name == 'bike':
                bikes.append([x1, y1, x2, y2])
            elif class_name == 'no helmet':
                no_helmets.append([x1, y1, x2, y2])
                violations.append("No Helmet")
            elif class_name == 'helmet':  # Added helmet detection
                helmets.append([x1, y1, x2, y2])
            elif class_name == 'mirror':
                mirrors.append([x1, y1, x2, y2])
            elif class_name == 'number plate':
                plate_boxes.append([x1, y1, x2, y2, conf])
        
        # Process each bike detection for mirror violations
        for bike_bbox in bikes:
            bike_center = [(bike_bbox[0] + bike_bbox[2])/2, (bike_bbox[1] + bike_bbox[3])/2]
            bike_width = bike_bbox[2] - bike_bbox[0]
            bike_height = bike_bbox[3] - bike_bbox[1]
            
            # Check for mirrors
            mirrors_near_bike = []
            for mirror_bbox in mirrors:
                mirror_center = [(mirror_bbox[0] + mirror_bbox[2])/2, 
                            (mirror_bbox[1] + mirror_bbox[3])/2]
                
                # Check if mirror is near the bike (in the upper portion)
                if (abs(mirror_center[0] - bike_center[0]) < bike_width and
                    mirror_center[1] > bike_bbox[1] and
                    mirror_center[1] < bike_center[1]):
                    mirrors_near_bike.append(mirror_bbox)
            
            # Only add mirror violation if no mirrors are detected near this bike
            if len(mirrors_near_bike) == 0:
                violations.append("Missing Rear-view Mirror")
        
        # Process plates for no_helmet detections
        for no_helmet_box in no_helmets:
            # Find nearest plate
            if plate_boxes:
                nearest_plate = min(plate_boxes, 
                                key=lambda x: abs((x[0] + x[2])/2 - (no_helmet_box[0] + no_helmet_box[2])/2))
                
                # Try to read the plate
                plate_text = self.read_license_plate(frame, nearest_plate)
                if plate_text and plate_text not in license_plates:
                    license_plates.append(plate_text)
        
        # Draw all detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = results.names[int(class_id)]
            
            # Set colors based on class
            if class_name == 'no helmet':
                color = (0, 0, 255)  # Red
            elif class_name == 'number plate':
                color = (255, 255, 0)  # Yellow
            elif class_name == 'mirror':
                color = (0, 255, 255)  # Cyan
            elif class_name == 'helmet':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 255, 0)  # Green for bike
            
            # Draw boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw violations and plates
        y_offset = 30
        # Remove duplicates while preserving order
        unique_violations = []
        [unique_violations.append(v) for v in violations if v not in unique_violations]
        
        for i, violation in enumerate(unique_violations):
            cv2.putText(frame, f"Violation {i+1}: {violation}", 
                    (10, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw detected plates
        plate_offset = y_offset + len(unique_violations)*30
        '''for i, plate in enumerate(license_plates):
            cv2.putText(frame, f"Violation Plate {i+1}: {plate}", 
                    (10, plate_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)'''
        
        return frame, results, unique_violations, license_plates
    
    def is_bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or 
                x1_min > x2_max or 
                y1_max < y2_min or 
                y1_min > y2_max)


    def process_video_optimized(self, video_path, display=True, progress_callback=None):
        """Process video with progress updates"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = 'output_detection.mp4'
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (width, height))
        
        frame_count = 0
        all_violations = []
        all_plates = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame, results, violations, plates = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Collect results
                all_violations.extend(violations)
                all_plates.extend(plates)
                
                # Update progress
                frame_count += 1
                if progress_callback:
                    progress = frame_count / total_frames
                    status = f"Processing frame {frame_count}/{total_frames}"
                    progress_callback(progress, status)
                
                # Display if requested
                if display:
                    cv2.imshow('Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except Exception as e:
            print(f"Error processing video: {e}")
            raise e
        
        finally:
            # Clean up
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Return unique violations and plates
        return output_path, list(set(all_violations)), list(set(all_plates))

    def _process_frame_batch(self, input_queue, output_queue):
        """Worker process for batch frame processing"""
        while True:
            frame = input_queue.get()
            if frame is None:
                break
                
            processed_frame, _, violations, plates = self.process_frame(frame)
            output_queue.put((processed_frame, violations, plates))

    def process_image(self, image_path):
        """Process image file"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        return self.process_frame(img)