# Black-Box_Testing_For_Autonomous_Driving

A full-stack web application that simulates an autonomous driving decision system. It combines a Machine Learning model for lane-based steering angle prediction with a YOLOv8 object detection system to make intelligent, real-time safety overrides (such as slowing down or stopping).

## ✨ Features

- **Hybrid AI Logic:** Merges traditional Machine Learning (Random Forest) for steering predictions with deep learning (YOLOv8) for dynamic object detection.
- **Robustness Testing:** Includes scripts to evaluate the model's performance under various weather and lighting conditions (rain, snow, night, fog).
- **Intelligent Decision Engine:** Analyzes object proximity using bounding box area ratios and intelligently adjusts the steering direction to avoid collisions.
- **Modern Interactive UI:** A sleek, responsive dashboard featuring visual indicators, a dynamic steering wheel animation, and real-time bounding boxes drawn over the uploaded scene.
- **FastAPI Backend:** High-performance REST API for processing images and serving the frontend interface seamlessly.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, OpenCV (cv2)
- **Machine Learning:** Scikit-Learn (Random Forest), Ultralytics (YOLOv8)
- **Frontend:** HTML5, Vanilla JavaScript, CSS3
- **Data Visualization:** Chart.js (for feature importance graphs)

## ⚙️ Installation & Setup

Follow these steps to run the application on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/harshit1525/Black-Box_Testing_For_Autonomous_Driving.git
cd Black-Box_Testing_For_Autonomous_Driving
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 5. Start the Server
```bash
uvicorn main:app --reload
```
The FastAPI server will start, and the frontend will be served automatically.

### 6. Open the Dashboard
Open your web browser and go to:
**[http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)**

## 🧠 How It Works (The Pipeline)

1. **Upload:** A user drags and drops a driving scene image into the frontend dashboard.
2. **Lane Detection:** The backend uses OpenCV to extract lanes and structural features from the image.
3. **Steering Prediction:** The pre-trained Random Forest model (`rf_model.joblib`) predicts a base steering angle (Left, Right, or Straight).
4. **Object Detection (Safety Override):** YOLOv8 scans the image for obstacles (cars, pedestrians). It calculates their position and estimated distance.
5. **Final Decision:** The logic engine merges the steering prediction with object proximity data. If an obstacle is too close, the system overrides the steering to **SLOW DOWN** or **STOP** and displays this reasoning on the UI.

---

> *This project was built to demonstrate how Black-Box Machine Learning models can be integrated with deterministic safety logic for autonomous vehicle testing.*
