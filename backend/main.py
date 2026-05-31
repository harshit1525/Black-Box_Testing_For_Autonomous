import os
import json
import joblib
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import base64
import vision_utils

app = FastAPI(title="Steering Angle Prediction API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from frontend folder
app.mount("/static", StaticFiles(directory="../frontend", html=True), name="static")

# Load Random Forest model and feature importances globally
MODEL_PATH = "rf_model.joblib"
IMPORTANCE_PATH = "feature_importance.json"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

if os.path.exists(IMPORTANCE_PATH):
    with open(IMPORTANCE_PATH, "r") as f:
        feature_importances = json.load(f)
else:
    feature_importances = {}

# Load YOLOv8 model for object detection
try:
    model_yolo = YOLO("yolov8n.pt")
except Exception as e:
    model_yolo = None
    print("Warning: YOLOv8 model failed to load.", e)


@app.get("/")
def read_root():
    return {"message": "Robust Steering Angle Prediction API is running!"}

@app.get("/feature-importance")
def get_feature_importance():
    if not feature_importances:
        raise HTTPException(status_code=404, detail="Feature importances not found.")
    return feature_importances

@app.post("/predict")
async def predict_steering(file: UploadFile = File(...), enable_od: bool = Form(False)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please run save_model.py first.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        # Detect lanes using vision pipeline
        lanes = vision_utils.detect_lanes(img)
        
        if not lanes or len(lanes) < 6:
            raise HTTPException(status_code=400, detail="Could not detect enough lane points in the image.")
            
        features = vision_utils.extract_features(lanes)
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract lane features (e.g. missing left/right lanes).")
            
        # Predict steering using the model
        X_input = np.array([features])
        prediction = model.predict(X_input)[0]
        
        # Determine initial lane-based direction
        direction = "straight"
        if prediction < -0.05:
            direction = "left"
        elif prediction > 0.05:
            direction = "right"
            
        objects_detected = []
        final_decision = direction
        
        # --- Object Detection & Logic Override ---
        decision_reason = "Path clear. Following lane prediction."
        distance_estimation = []
        steer_dir = direction
        
        if enable_od and model_yolo:
            results = model_yolo(img, verbose=False)
            img_h, img_w = img.shape[:2]
            img_area = img_h * img_w
            
            # COCO indices: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
            relevant_classes = [0, 2, 3, 5, 7]
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id in relevant_classes and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model_yolo.names[cls_id]
                        
                        # Determine position
                        center_x = (x1 + x2) / 2
                        if center_x < img_w / 3:
                            position = "left"
                        elif center_x > 2 * img_w / 3:
                            position = "right"
                        else:
                            position = "center"
                            
                        # Distance estimation
                        bbox_area = (x2 - x1) * (y2 - y1)
                        area_ratio = bbox_area / img_area
                        if area_ratio > 0.15:
                            distance = "close"
                        elif area_ratio > 0.05:
                            distance = "medium"
                        else:
                            distance = "far"
                            
                        distance_estimation.append({
                            "label": label,
                            "position": position,
                            "distance": distance
                        })
                            
                        objects_detected.append({
                            "label": label,
                            "confidence": conf,
                            "position": position,
                            "distance": distance,
                            "bbox": [x1, y1, x2 - x1, y2 - y1] # x, y, width, height
                        })
                        
            # Decision Logic - Distance & Priority Based
            is_close_center = any(obj["distance"] == "close" and obj["position"] == "center" for obj in distance_estimation)
            is_medium_center = any(obj["distance"] == "medium" and obj["position"] == "center" for obj in distance_estimation)
            
            is_close_left = any(obj["distance"] == "close" and obj["position"] == "left" for obj in distance_estimation)
            is_medium_left = any(obj["distance"] == "medium" and obj["position"] == "left" for obj in distance_estimation)
            
            is_close_right = any(obj["distance"] == "close" and obj["position"] == "right" for obj in distance_estimation)
            is_medium_right = any(obj["distance"] == "medium" and obj["position"] == "right" for obj in distance_estimation)
            
            has_close = any(obj["distance"] == "close" for obj in distance_estimation)
            has_medium = any(obj["distance"] == "medium" for obj in distance_estimation)
            
            # Priority 1: Center is CLOSE or both sides are CLOSE
            if is_close_center or (is_close_left and is_close_right):
                final_decision = "stop"
                decision_reason = "Obstacle VERY CLOSE → STOP"
                steer_dir = "straight"
            # Priority 2: Center is MEDIUM
            elif is_medium_center:
                final_decision = "slow down"
                decision_reason = "Obstacle near CENTER → slowing down"
                steer_dir = "straight"
            else:
                # 1. Base Scores
                if direction == "left":
                    score_left, score_straight, score_right = 10, 5, 2
                elif direction == "right":
                    score_left, score_straight, score_right = 2, 5, 10
                else:
                    score_left, score_straight, score_right = 2, 10, 2
                    
                # 2. Penalties
                penalty_left = 50 if is_close_left else (15 if is_medium_left else 0)
                penalty_right = 50 if is_close_right else (15 if is_medium_right else 0)
                
                # 3. Final Scores
                final_left = score_left - penalty_left
                final_straight = score_straight - 0
                final_right = score_right - penalty_right
                
                # 4. Make Decision
                max_score = max(final_left, final_straight, final_right)
                
                if max_score < 0:
                    steer_dir = "straight"
                    if has_medium:
                        final_decision = "slow down"
                        decision_reason = "Obstacles on all sides (Medium) → SLOW DOWN"
                    else:
                        final_decision = "stop"
                        decision_reason = "Obstacles blocking all paths → STOP"
                else:
                    if final_left == max_score and final_left >= final_right and final_left >= final_straight:
                        steer_dir = "left"
                    elif final_right == max_score and final_right >= final_left and final_right >= final_straight:
                        steer_dir = "right"
                    else:
                        steer_dir = "straight"
                        
                    # 5. Formulate reasoning
                    if steer_dir != direction:
                        if is_medium_left or is_medium_right or is_close_left or is_close_right:
                            final_decision = "slow down"
                            decision_reason = f"Obstacle nearby → adjusting {steer_dir.upper()} and SLOW DOWN"
                        else:
                            final_decision = steer_dir
                            decision_reason = f"Adjusted to {steer_dir.upper()} for safety."
                    else:
                        if is_medium_left or is_medium_right or is_close_left or is_close_right:
                            final_decision = "slow down"
                            decision_reason = f"Lane clear but obstacle nearby → SLOW DOWN"
                        else:
                            final_decision = steer_dir
                            decision_reason = "All objects far. Following lane prediction."
                
        return {
            "steering_angle": float(prediction),
            "lane_direction": direction,
            "direction": steer_dir,
            "detected_lanes": len(lanes),
            "objects_detected": objects_detected,
            "distance_estimation": distance_estimation,
            "final_decision": final_decision,
            "decision_reason": decision_reason
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
