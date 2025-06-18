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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
KNOWN_WIDTH = 0.1       # Width in meters of known object
FOCAL_LENGTH = 600      # Approximate focal length of your camera
TARGET_FPS = 60         # Target camera FPS
FRAME_SKIP = 30         # Frame skip interval for detection

def calculate_distance(pixel_width):
    """Estimate distance based on object's pixel width."""
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

@st.cache_resource
def load_model():
    model_path = Path("weights/yolov8n.pt")
    model = YOLO(model_path)
    model.conf = 0.25
    model.iou = 0.45
    return model

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎥 Real-Time Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect and localize objects using YOLOv8 in live webcam feed</p>", unsafe_allow_html=True)
st.markdown("---")
st.title("SensAura: Live Object Detection and Navigation")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("🎯 Detection Confidence", 0.0, 1.0, 0.25, 0.01)
    frame_skip = st.slider("🎞️ Frame Skip Interval", 1, 60, FRAME_SKIP)
    start_button = st.button("🚀 Start Camera")
    stop_button = st.button("🛑 Stop Camera")


# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Toggle camera state
if start_button:
    st.session_state.camera_running = True
if stop_button:
    st.session_state.camera_running = False


# Load model
model = load_model()

# UI placeholders
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
info_placeholder = col2.container()
fps_placeholder = col2.empty()

# Start camera and detection loop
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    prev_time = time.time()
    frame_count = 0
    fps = 0
    frame_skip_counter = 0
    annotated_frame = None

    try:
        while cap.isOpened() and st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Camera disconnected or video ended.")
                break

            # FPS tracking
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = current_time

            frame_skip_counter += 1

            if frame_skip_counter >= frame_skip:
                frame_skip_counter = 0
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()

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

                # Display detection results
                if detections:
                    with info_placeholder:
                        st.subheader("🔍 Detected Objects")
                        tts_lines = []
                        for det in detections:
                            direction = (
                                "Left" if det['x_center'] < 0.33 else
                                "Right" if det['x_center'] > 0.66 else
                                "Center"
                            )
                            proximity = (
                                "🟥 Close" if det['distance'] < 2 else
                                "🟨 Medium" if det['distance'] < 5 else
                                "🟩 Far"
                            )
                            st.markdown(
                                f"- **{det['class']}** {direction} | {proximity} | Confidence: `{det['confidence']:.2f}`"
                            )
                            tts_lines.append(f"{det['class']} {direction} is {proximity}")
                        get_speak_worker().speak(". ".join(tts_lines))
                else:
                    info_placeholder.info("No objects detected.")

            # Display current frame
            display_frame = annotated_frame if annotated_frame is not None else frame
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_placeholder.image(display_frame, channels="BGR", use_column_width=True)
            fps_placeholder.markdown(f"### ⏱️ Current FPS: `{fps}`")

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()
