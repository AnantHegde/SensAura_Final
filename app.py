# Python In-built packages
import os
import sys
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')

# Set environment variables before any imports
os.environ["YOLO_DISABLE_HUB"] = "1"
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['ULTRA_DISABLE_SIGNAL_HANDLING'] = '1'
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Apply patch before importing ultralytics
import patch_ultralytics

# External packages
import streamlit as st
import cv2
from pathlib import Path
from ultralytics import YOLO

# Local Modules
import settings
import helper
from helper import get_speak_worker

# Configure Streamlit page
st.set_page_config(
    page_title="Live Object Detection",
    page_icon="🎥",
    layout="wide"
)

# Constants for distance estimation
KNOWN_WIDTH = 0.1  # Known width of objects in meters (adjust based on your use case)
FOCAL_LENGTH = 500  # Approximate focal length of the camera (adjust based on your camera)
TARGET_FPS = 60
FRAME_SKIP = 1  # Process every nth frame

def calculate_distance(pixel_width):
    """Calculate distance using the pinhole camera model"""
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

# Load YOLO model
@st.cache_resource
def load_model():
    # Use the smallest and fastest model
    model_path = Path("weights/yolov8n.pt")
    model = YOLO(model_path)
    # Set model parameters for speed
    model.conf = 0.25  # Lower confidence threshold for faster processing
    model.iou = 0.45   # Lower IOU threshold for faster processing
    return model

# Main app
st.title("Live Object Detection")

# Load model
model = load_model()

# Confidence threshold
confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25)

# Create placeholders for information display
info_placeholder = st.empty()
frame_placeholder = st.empty()
fps_placeholder = st.empty()

# Start camera
if st.sidebar.button("Start Camera"):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set higher resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # Initialize FPS calculation variables
    prev_time = time.time()
    frame_count = 0
    fps = 0
    frame_skip_counter = 0
    
    try:
        # Initialize variables for the loop
        annotated_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video stream ended or camera disconnected.")
                break

            # --- FPS Calculation ---
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = current_time

            frame_skip_counter += 1
            
            # --- Detection and Annotation (on specified frames) ---
            if frame_skip_counter >= FRAME_SKIP:
                frame_skip_counter = 0
                
                # Run detection
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()  # Update the annotated frame
                
                # Process detections for info and TTS
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'class': model.names[int(box.cls[0].cpu().numpy())],
                            'distance': calculate_distance(x2 - x1),
                            'confidence': float(box.conf[0].cpu().numpy()),
                            'x_center': (x1 + x2) / 2 / frame.shape[1]
                        })

                # --- Update UI Text and TTS ---
                if detections:
                    info_text = "Detected Objects:\n"
                    tts_lines = []
                    for det in detections:
                        # Calculate direction
                        x_center = det.get('x_center', 0.5)
                        direction = "in the center"
                        if x_center < 0.33: direction = "on the left"
                        elif x_center > 0.66: direction = "on the right"
                        
                        # Proximity label
                        distance = det['distance']
                        proximity = "far"
                        if distance < 2: proximity = "close"
                        elif distance < 5: proximity = "medium"

                        line = f"{det['class']} {direction} is {proximity}"
                        info_text += f"- {line}\n"
                        tts_lines.append(line)
                    
                    info_placeholder.text(info_text)
                    tts_text = ". ".join(tts_lines)
                    get_speak_worker().speak(tts_text)
                else:
                    info_placeholder.text("No objects detected")

            # --- Frame Display (always run) ---
            # Use the latest annotated_frame, or the raw frame if none exists yet
            display_frame = annotated_frame if annotated_frame is not None else frame
            
            # Add FPS text to the frame that will be displayed
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            
            # Display FPS separately for clarity
            fps_placeholder.text(f"Current FPS: {fps}")

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
    finally:
        if cap.isOpened():
            cap.release()
