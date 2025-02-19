# Real-Time Robotic Arm Control using OpenCV & Arduino
## 🔥 Overview
This project implements a **real-time robotic arm control system** using **OpenCV** for object tracking and hand gesture recognition. It processes image data to determine object positions and translates them into servo motor commands via **Arduino**, enabling precise robotic arm movement.

## 🚀 Features
- **Object Detection & Tracking**: Uses **HSV thresholding, Canny Edge Detection, and Hough Transform** for accurate object recognition.
- **Hand Gesture Control**: Haar cascade and hand gesture classifiers enable intuitive control.
- **Real-Time Processing**: Offloads computer vision tasks to a **PC** for better performance.
- **PID-Based Servo Control**: Ensures smooth and precise robotic arm movement.
- **Efficient Serial Communication**: Uses **PySerial** for low-latency data transmission.

## 🛠️ Hardware Requirements
- **Microcontroller**: Arduino Uno
- **Camera Module**: USB Webcam
- **Servo Motors**: MG995 / SG90
- **Robotic Arm Kit**: 4-DOF/5-DOF (3D printed or metal-based)
- **Power Supply**: 5V/6V DC for servo motors
- **Computer**: Runs OpenCV for image processing

## 💻 Software Requirements
- **Programming Languages**: Python (for OpenCV), C++ (for Arduino)
- **Libraries**: OpenCV, NumPy, PySerial, CVZone (for AI object detection)
- **Development Tools**: Arduino IDE, VS Code
- **Operating System**: Windows/Linux

## ⚙️ Workflow
1. **Capture Image/Video** using OpenCV.
2. **Preprocess Image** (HSV conversion, thresholding).
3. **Detect Object Features** (Edge detection, contour filtering).
4. **Track Object Position & Orientation**.
5. **Send Processed Data to Arduino** via PySerial.
6. **Calculate Servo Angles** using Inverse Kinematics & PID Control.
7. **Move Robotic Arm Dynamically** to pick and place objects.
8. **Loop Until Stop Condition**.

## 🎯 Applications
- Assistive robotics for accessibility solutions.
- Industrial automation and object sorting.
- Human-robot interaction for real-time control.
- Gesture-controlled robotic systems.

