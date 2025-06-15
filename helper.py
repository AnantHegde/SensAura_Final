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

# Apply patch before importing ultralytics
import patch_ultralytics

import streamlit as st
import cv2
import yt_dlp
import settings
from yolo_model import YOLOModel
from ultralytics import YOLO
import threading
import queue
import time
import pyttsx3
from typing import Optional

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None

class TTSEngineManager:
    """Thread-safe manager for pyttsx3 engine with better error handling."""
    
    def _init_(self):
        self._engine_lock = threading.Lock()
        self._engine = None
        self._active = False
        
    def get_engine(self) -> Optional[pyttsx3.Engine]:
        """Get or create a TTS engine instance."""
        with self._engine_lock:
            if self._engine is None:
                try:
                    self._engine = pyttsx3.init()
                    self._engine.setProperty('rate', 150)
                    self._engine.setProperty('volume', 0.9)
                    self._active = True
                except Exception as e:
                    st.error(f"Failed to initialize TTS engine: {e}")
                    return None
            return self._engine
        
    def shutdown(self):
        """Clean up the engine."""
        with self._engine_lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                    self._engine = None
                    self._active = False
                except Exception as e:
                    st.error(f"Error shutting down TTS engine: {e}")

class SpeakWorker:
    """Improved background thread for TTS with better queue management."""
    
    def _init_(self, cooldown: float = 5.0):
        self.cooldown = cooldown
        self._queue = queue.Queue()
        self._last_text = ""
        self._last_time = 0.0
        self._tts_manager = TTSEngineManager()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def speak(self, text: str):
        """Enqueue text for speaking (non-blocking)."""
        if text and self._running:
            self._queue.put(text)
            
    def _run(self):
        """Main worker loop."""
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
                now = time.time()
                
                # Skip if same as last and within cooldown
                if text == self._last_text and (now - self._last_time) < self.cooldown:
                    continue
                    
                # Get engine and speak
                engine = self._tts_manager.get_engine()
                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                        self._last_text = text
                        self._last_time = time.time()
                    except Exception as e:
                        st.warning(f"TTS error: {e}")
                        # Attempt to recover by getting a fresh engine
                        self._tts_manager.shutdown()
                        engine = self._tts_manager.get_engine()
                        if engine:
                            try:
                                engine.say(text)
                                engine.runAndWait()
                                self._last_text = text
                                self._last_time = time.time()
                            except Exception as e2:
                                st.warning(f"TTS recovery failed: {e2}")
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Unexpected error in SpeakWorker: {e}")
                
    def shutdown(self):
        """Clean up the worker thread."""
        self._running = False
        self._tts_manager.shutdown()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

@st.cache_resource
def get_speak_worker():
    """Returns a cached SpeakWorker instance with proper cleanup."""
    worker = SpeakWorker()
    # Register cleanup handler
    import atexit
    atexit.register(worker.shutdown)
    return worker

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    if res is None:
        st.error("Model prediction failed")
        return

    # Announce detected objects
    detected_classes = []
    if hasattr(res[0], 'boxes') and res[0].boxes is not None and len(res[0].boxes) > 0:
        unique_cls_ids = sorted(list(set(res[0].boxes.cls.int().tolist())))
        
        for cls_id in unique_cls_ids:
            class_name = res[0].names[cls_id]
            detected_classes.append(class_name)

    if detected_classes:
        if len(detected_classes) == 1:
            detected_text = f"Detected {detected_classes[0]}."
        elif len(detected_classes) == 2:
            detected_text = f"Detected {detected_classes[0]} and {detected_classes[1]}."
        else:
            first_items = ", ".join(detected_classes[:-1])
            detected_text = f"Detected {first_items}, and {detected_classes[-1]}."
        
        # Get the speak worker and make it speak
        speak_worker = get_speak_worker()
        speak_worker.speak(detected_text)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True)

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            stream_url = get_youtube_stream_url(source_youtube)
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error("Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker
                    )
                else:
                    break
            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")

def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                           model,
                                           st_frame,
                                           image,
                                           is_display_tracker,
                                           tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))

def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    st_frame = st.empty()
    
    if st.sidebar.button('Start Detection'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model.predict(image, conf=conf, verbose=False)
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted,
                                caption='Detected Video',
                                channels="BGR",
                                use_container_width=True)
                else:
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        finally:
            if 'vid_cap' in locals():
                vid_cap.release()

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                           model,
                                           st_frame,
                                           image,
                                           is_display_tracker,
                                           tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def main():
    st.title("Object Detection with YOLOv8")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model Options
    model_path = str(settings.DETECTION_MODEL)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.error("Model failed to load. Exiting...")
        return
    
    # Confidence slider
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    
    # Application selection
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["YouTube", "RTSP Stream", "Webcam", "Video"])
    
    if app_mode == "YouTube":
        play_youtube_video(confidence, model)
    elif app_mode == "RTSP Stream":
        play_rtsp_stream(confidence, model)
    elif app_mode == "Webcam":
        play_webcam(confidence, model)
    elif app_mode == "Video":
        play_stored_video(confidence, model)

if _name_ == "_main_":
    main()
