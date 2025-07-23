# Sensaura - Privacy-First Intelligence for Human Navigation

## a. Application Description

Sensaura is a real-time object detection application built using Streamlit and YOLOv8. It allows users to detect objects in live video streams from webcams, RTSP streams. The application provides a user-friendly interface to adjust detection confidence and displays the detected objects with bounding boxes, along with textual information and Text-to-Speech (TTS) output for accessibility.

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

## e. Demo of SensAura in action:

Video Link: https://drive.google.com/file/d/1bXwdf-moJ51DA9zMVF3RteNJnjS245Gk/view?usp=sharing

## f. Future advancements using android app:

This mobile app is an extra showcase to demonstrate the potential of SensAura AI if implemented on a mobile platform. It is not part of the official hackathon submission for the Edge AI Developer Hackathon (June 14-15, 2025, Qualcomm Bengaluru), as it runs on a mobile device (not a Snapdragon X Elite-powered Copilot+ PC) and uses cloud-based Gemini APIs, which do not comply with the hackathon’s edge AI requirements. Our official submission focuses on live image processing and voice output implemented on the provided laptop. This README is provided to illustrate additional features (live object detection for navigation and fall detection with emergency SMS) that could enhance SensAura AI’s vision for assisting visually impaired and dementia-affected individuals.

Video Demo of working app: https://drive.google.com/file/d/1FMOnMlKLOvGjhodKRlg-YCvSXn1-f4Ei/view?usp=sharing


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
Adjust the KNOWN_WIDTH and FOCAL_LENGTH constants in app.py for accurate distance estimation.
The application uses yolov8n.pt by default, you can change the model in settings.py.
