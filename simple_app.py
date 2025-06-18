import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set environment variables before any imports
os.environ["YOLO_DISABLE_HUB"] = "1"
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['ULTRA_DISABLE_SIGNAL_HANDLING'] = '1'
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np
# pyttsx3, threading, queue removed as helper.py handles TTS
import nest_asyncio
from helper import _display_detected_frames # Use helper for frame processing

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


# Set page config
st.set_page_config(
    page_title="Live Object Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Live Object Detection")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("weights/yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()
if model is None:
    st.error("Failed to load model. Please check if the model file exists.")
    st.stop()

# Confidence slider
confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, key="confidence_slider")


# Main detection loop using helper function
def run_detection():
    frame_placeholder = st.empty() # For displaying video frames
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open camera")
        return
    
    try:
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from camera")
                break
            
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                # Call helper function to process and display frame, and handle TTS
                # The helper function will resize the frame as needed.
                _display_detected_frames(confidence, model, frame_placeholder, frame)
            else:
                # Still display the frame to keep the video feed smooth, but don't process it
                # Resize to a standard display size if not already done by helper or if helper is skipped
                # The helper _display_detected_frames resizes to 720x(720*9/16) = 720x405
                # For skipped frames, we should display them without YOLO overlay.
                # To keep it simple, we can just show the raw frame or a resized version.
                # However, _display_detected_frames handles the st_frame.image update.
                # A simple way is to pass the frame to a lightweight display function or just update the placeholder with the raw frame.
                # For now, let's ensure the placeholder is updated even for skipped frames to avoid stale image.
                # We'll resize it to what _display_detected_frames would do for consistency in display size.
                display_frame = cv2.resize(frame, (720, int(720*(9/16))))
                frame_placeholder.image(display_frame, channels="BGR")

            # Add a small delay to control frame rate and allow UI updates
            time.sleep(0.01) 
            
    except Exception as e:
        st.error(f"Error in detection loop: {str(e)}")
        # For more detailed debugging, print traceback to console
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        # The SpeakWorker in helper.py is a daemon thread and manages its own lifecycle;
        # no explicit stop action (like a poison pill) is needed from here.

# Start camera button
if st.sidebar.button("Start Camera"):
    run_detection()
