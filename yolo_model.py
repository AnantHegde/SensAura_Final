import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix 1: Disable Ultralytics Hub
os.environ["YOLO_DISABLE_HUB"] = "1"
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['ULTRA_DISABLE_SIGNAL_HANDLING'] = '1'

# Apply the ultralytics patch
import patch_ultralytics

import streamlit as st
from pathlib import Path

# Fix 2: Lazy load YOLO with caching
@st.cache_resource
def get_yolo_model(model_path):
    try:
        # Verify model path exists
        model_path = Path(model_path)
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            return None
            
        print(f"Loading model from: {model_path}")
        from ultralytics import YOLO
        model = YOLO(str(model_path), task='detect')
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Detailed error: {str(e)}")
        return None

class YOLOModel:
    def _init_(self, model_path):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            print(f"Initializing model with path: {self.model_path}")
            self.model = get_yolo_model(self.model_path)
            if self.model is None:
                print("Model initialization failed")
            else:
                print("Model initialized successfully")
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            self.model = None

    def predict(self, image, conf=0.25):
        if self.model is None:
            print("Cannot predict: model is None")
            return None
        try:
            return self.model.predict(image, conf=conf)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

    def track(self, image, conf=0.25, persist=True, tracker=None):
        if self.model is None:
            print("Cannot track: model is None")
            return None
        try:
            return self.model.track(image, conf=conf, persist=persist, tracker=tracker)
        except Exception as e:
            print(f"Tracking error: {str(e)}")
            return None
