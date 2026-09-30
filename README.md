# AI Surveillance using YOLOv8

A real-time computer vision project built with YOLOv8 and OpenCV for webcam-based object detection. The project includes a general object-detection mode and a focused surveillance mode for detecting people and common bag/luggage classes.

## Features

- Real-time object detection from a webcam
- YOLOv8 Nano (`yolov8n.pt`) inference
- OpenCV-based video capture and display
- General object detection with `detect.py`
- Focused surveillance detection with `bag_detector.py`
- Detects `person`, `backpack`, `handbag`, and `suitcase` in the focused mode
- Press `q` to exit the live detection window

## Project Structure

```text
ai-surveillance-yolov8/
├── bag_detector.py    # Focused surveillance detector
├── detect.py          # General YOLOv8 webcam detection
├── yolov8n.pt         # YOLOv8 Nano pretrained model
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- Python
- Ultralytics YOLOv8
- OpenCV

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure your webcam is available.

## Run

For general object detection:

```bash
python detect.py
```

For the focused surveillance/bag detector:

```bash
python bag_detector.py
```

Press **q** in the OpenCV window to stop the application.

## How it works

The application captures frames from the webcam using OpenCV and passes each frame to a pretrained YOLOv8 Nano model. The model returns detected objects and their bounding boxes. The general detector displays the model's detections, while the focused detector filters detections to people and selected bag/luggage classes before drawing labels and bounding boxes.

## Notes

- This project is a local computer-vision prototype using a pretrained YOLOv8 model.
- The repository does not include training data or a custom-trained model.
- Detection quality depends on lighting, camera quality, distance, and the capabilities of the pretrained model.
