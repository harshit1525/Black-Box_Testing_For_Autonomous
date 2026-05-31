import json
import cv2
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

json_path = "../bdd100k_labels_images_train.json"
image_folder = "../dataset"

print("Loading dataset metadata...")
with open(json_path) as f:
    data = json.load(f)

# ---------------- STEERING ---------------- #

def compute_steering(img, lanes):
    xs = [p[0] for p in lanes]
    lane_center = np.mean(xs)
    image_center = img.shape[1] / 2
    return (lane_center - image_center) / image_center

# ---------------- FEATURE EXTRACTION ---------------- #

def extract_features(lanes):
    xs = np.array([p[0] for p in lanes])
    ys = np.array([p[1] for p in lanes])

    if len(xs) < 6:
        return None

    # Split into left and right lanes
    mid = np.median(xs)
    left_x = xs[xs < mid]
    right_x = xs[xs >= mid]

    if len(left_x) < 2 or len(right_x) < 2:
        return None

    # Core features
    left_mean = float(np.mean(left_x))
    right_mean = float(np.mean(right_x))
    lane_width = float(right_mean - left_mean)

    # Curvature approximation
    curvature = float(np.std(xs))

    # Vertical distribution
    y_spread = float(np.std(ys))

    return [left_mean, right_mean, lane_width, curvature, y_spread]

# ---------------- DATASET PREPARATION ---------------- #

X = []
y = []

count = 0
print("Extracting features from images...")
for item in data:
    img_name = item["name"]
    img_path = os.path.join(image_folder, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None: continue

    lanes = []
    for label in item["labels"]:
        if label["category"] == "lane":
            poly = label["poly2d"][0]["vertices"]
            for p in poly:
                lanes.append(p)

    if len(lanes) < 6:
        continue

    features = extract_features(lanes)
    if features is None:
        continue

    steering = compute_steering(img, lanes)

    X.append(features)
    y.append(steering)

    count += 1
    if count == 1000:   # Limit to 1000 samples for training speed
        break

print("Dataset size:", len(X))

X = np.array(X)
y = np.array(y)

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ---------------- #

print("Training Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Saving model to rf_model.joblib...")
joblib.dump(model, "rf_model.joblib")

# Save feature importances
feature_names = ["left_mean", "right_mean", "lane_width", "curvature", "y_spread"]
importances = model.feature_importances_.tolist()
importance_dict = dict(zip(feature_names, importances))

with open("feature_importance.json", "w") as f:
    json.dump(importance_dict, f)

print("Done! Model and feature importances saved.")
