import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

json_path = "bdd100k_labels_images_train.json"
image_folder = "dataset"

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
    left_mean = np.mean(left_x)
    right_mean = np.mean(right_x)
    lane_width = right_mean - left_mean

    # Curvature approximation
    curvature = np.std(xs)

    # Vertical distribution
    y_spread = np.std(ys)

    return [left_mean, right_mean, lane_width, curvature, y_spread]

# ---------------- DATASET ---------------- #

X = []
y = []

count = 0

for item in data:

    img_name = item["name"]
    img_path = os.path.join(image_folder, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
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
    if count == 800:   # more data = better
        break

print("Dataset size:", len(X))

X = np.array(X)
y = np.array(y)

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ---------------- #

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
# ---------------- FEATURE IMPORTANCE ---------------- #

import numpy as np
import matplotlib.pyplot as plt

feature_names = [
    "left_mean",
    "right_mean",
    "lane_width",
    "curvature",
    "y_spread"
]

importances = model.feature_importances_

# Print values
for name, val in zip(feature_names, importances):
    print(f"{name}: {val:.4f}")

# Plot
plt.figure(figsize=(8,5))
plt.bar(feature_names, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ---------------- PREDICTION ---------------- #

y_pred = model.predict(X_test)

# ---------------- EVALUATION ---------------- #

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# ---------------- ROBUSTNESS ANALYSIS ---------------- #

print("\n--- ROBUSTNESS ANALYSIS ---")

noise_levels = [0, 3, 6, 10]

robust_mae = []

for noise in noise_levels:

    X_noisy = []
    y_noisy = []

    for item in data[:200]:   # small subset for testing

        img_name = item["name"]
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        lanes = []

        for label in item["labels"]:
            if label["category"] == "lane":
                poly = label["poly2d"][0]["vertices"]
                for p in poly:
                    lanes.append(p)

        if len(lanes) < 6:
            continue

        # Apply noise
        noisy_lanes = perturb_lanes(lanes, noise_level=noise)

        features = extract_features(noisy_lanes)
        if features is None:
            continue

        steering = compute_steering(img, noisy_lanes)

        X_noisy.append(features)
        y_noisy.append(steering)

    if len(X_noisy) == 0:
        continue

    X_noisy = np.array(X_noisy)
    y_noisy = np.array(y_noisy)

    y_pred_noisy = model.predict(X_noisy)

    mae_noisy = mean_absolute_error(y_noisy, y_pred_noisy)
    robust_mae.append(mae_noisy)

    print(f"Noise Level {noise}: MAE = {mae_noisy:.4f}")

# ---------------- PLOT ---------------- #

plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("Steering Angle Prediction (Final ML)")
plt.xlabel("Samples")
plt.ylabel("Steering Angle")
plt.savefig("ml_final.png")
plt.show()
print(model.feature_importances_)