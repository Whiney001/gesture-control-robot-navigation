# Gesture-Controlled Robot Navigation

A real-time vision-based control system that recognizes hand gestures and translates them into robot movement commands. Built using OpenCV and MediaPipe to enable intuitive hands-free interaction for robotics applications.

## Features
• Real-time gesture recognition pipeline  
• Detects multiple hand gestures (fist, open palm, peace, thumbs-up, point)  
• Maps gestures to movement actions (move, stop, turn, control speed)  
• Modular architecture for detection, classification, and control logic  
• Designed for integration with physical robot hardware

## Technologies Used
• Python  
• OpenCV  
• MediaPipe  
• NumPy

## Key Learning Outcomes
• Real-time computer vision processing  
• Feature extraction and gesture classification  
• Efficient frame processing and latency handling  
• Modular robotics-focused system design

## Run Instructions
```bash
pip install opencv-python mediapipe numpy
python gesture_recognition.py
