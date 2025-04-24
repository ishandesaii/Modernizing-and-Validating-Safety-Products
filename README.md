# 🔒 Modernizing and Validating Safety Products

This project is a comprehensive safety monitoring system integrating facial recognition, safety gear detection, smoke/fire detection, and intrusion alerts. It is designed to revolutionize safety practices across industrial and high-risk environments through automation, real-time analytics, and intelligent alerting systems.

---

## 🎯 Project Objective

To enhance workplace and site safety using Yolov8 and computer vision technologies by:
- Detecting the presence (or absence) of mandated personal protective equipment (PPE)
- Identifying smoke and fire threats early
- Monitoring unauthorized access via facial recognition
- Providing centralized control and alert management via an admin dashboard

---

## 🧠 Key Technologies

- **YOLOv8** for real-time object detection (helmets, vests, etc.)
- **CNN-based models** for face recognition and smoke/fire identification
- **Python**, **OpenCV**, and **TensorFlow/PyTorch** for model training and inference
- **Roboflow Datasets** for annotated PPE images
- Admin Dashboard for system monitoring and alerts

---

## 🧩 Features

- 🎥 **Safety Gear Detection**: Real-time alerts for PPE compliance (helmet, vest, etc.)
- 🔥 **Smoke & Fire Detection**: Early warning system for potential hazards
- 🚷 **Trespassing Detection**: Alerts for unauthorized access in sensitive zones
- 🧑‍💼 **Facial Recognition**: Secure and automatic identity verification
- 📊 **Dashboard**:
  - Custom camera zone configuration
  - Real-time alert display
  - Historical data access and report logs

---

## 🧪 Implementation Highlights

- **Model Training**: Used Roboflow datasets and annotated YOLOv8 images for safety gear detection.
- **Real-time Monitoring**: Implemented edge-based video analytics for speed and privacy.
- **YOLO format to pixel conversion**: Included utilities like `yolo2pixel()` for detection visualization.
- **Deployment**: Models packaged in `model.zip` for easy integration into production.

---

## 🚀 Future Scope

- Integrating IoT sensors for smarter responses
- AI-driven prediction models for hazard risk assessment
- Expanding coverage to gas leak detection, biometric authentication, etc.
- Cloud-based dashboard with mobile support

