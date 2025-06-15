# Sensaura - Live Object Detection Application

## a. Application Description

Sensaura is a real-time object detection application built using Streamlit and YOLOv8. It allows users to detect objects in live video streams from webcams, RTSP streams, YouTube videos, or stored video files. The application provides a user-friendly interface to adjust detection confidence and displays the detected objects with bounding boxes, along with textual information and Text-to-Speech (TTS) output for accessibility.

## b. Team Members

*   Anant Nagaraj Hegde - anhunchalli@gmail.com
*   Amruthesh C Hiremath - amrutheshchiremathbtech24@rvu.edu.in
*   Omkar Suresh Naik - omkarsnaik234@gmail.com
*   Nikhil Sridhara - Nikhilsridhara098@gmail.com

## c. Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd ananthegde-sensaura_final
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *   This command installs all the necessary Python packages listed in the `requirements.txt` file, including `streamlit`, `opencv-python`, `ultralytics`, and other dependencies.

4.  **Install system packages (if required):**

    *   The `packages.txt` file lists system-level dependencies. These can be installed using:

    ```bash
    sudo apt update
    sudo apt install -y $(cat packages.txt)
    ```

    *   This step is crucial for some environments, especially when dealing with OpenCV and its dependencies.

## d. Run and Usage Instructions

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    *   This command starts the Streamlit application, which will open in your default web browser.

2.  **Using the application:**

    *   **Sidebar Options:**
        *   **Detection Confidence:** Adjust the confidence threshold using the slider to control the sensitivity of object detection.
        *   **Start Camera:** Click this button to start object detection using your webcam.
    *   **Live Object Detection:**
        *   The main panel displays the live video feed with bounding boxes around detected objects.
        *   The application provides textual information about detected objects, including class names, proximity, and direction.
        *   Text-to-Speech (TTS) output announces the detected objects.
    *   **Other Modes (in `helper.py` and `simple_app.py`):**
        *   **YouTube:** Enter a YouTube video URL to detect objects in the video stream.
        *   **RTSP Stream:** Enter an RTSP stream URL to detect objects in the live stream.
        *   **Video:** Choose a stored video file to detect objects.

## Architecture

```mermaid
sequenceDiagram
    participant User
    participant StreamlitApp
    participant OpenCV
    participant YOLOModel
    participant TTS
    User->>StreamlitApp: Interacts with UI (e.g., clicks 'Start Camera')
    StreamlitApp->>OpenCV: Captures video frame (cv2.VideoCapture)
    OpenCV->>StreamlitApp: Returns video frame
    StreamlitApp->>YOLOModel: Performs object detection (model.predict)
    YOLOModel->>StreamlitApp: Returns detection results
    StreamlitApp->>StreamlitApp: Processes detection results (bounding boxes, labels)
    StreamlitApp->>TTS: Sends detected object information for speech
    TTS->>User: Announces detected objects
    StreamlitApp->>StreamlitApp: Displays annotated frame with bounding boxes
    StreamlitApp->>User: Updates UI with video and information


Notes:
Ensure your webcam is properly connected and accessible.
For RTSP streams, verify the URL and credentials.
Adjust the KNOWN_WIDTH and FOCAL_LENGTH constants in app.py for accurate distance estimation.
The application uses yolov8n.pt by default, you can change the model in settings.py.
