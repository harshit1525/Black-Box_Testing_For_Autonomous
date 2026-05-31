import cv2
import numpy as np

def detect_lanes(img):
    """
    Given a raw BGR image, this function applies an OpenCV pipeline to
    detect lane points so we can feed them into the feature extractor.
    This replaces the need for JSON labels when predicting on arbitrary images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    # Region of interest (lower half of the image)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2 + 50),
        (0, height // 2 + 50)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    lanes = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Sample points along the line
            num_points = 5
            for i in range(num_points):
                t = i / (num_points - 1)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                lanes.append([x, y])
    
    return lanes

def compute_steering(img, lanes):
    if len(lanes) == 0:
        return 0.0
    xs = [p[0] for p in lanes]
    lane_center = np.mean(xs)
    image_center = img.shape[1] / 2
    return (lane_center - image_center) / image_center

def extract_features(lanes):
    if len(lanes) < 6:
        return None

    xs = np.array([p[0] for p in lanes])
    ys = np.array([p[1] for p in lanes])

    mid = np.median(xs)
    left_x = xs[xs < mid]
    right_x = xs[xs >= mid]

    if len(left_x) < 2 or len(right_x) < 2:
        return None

    left_mean = float(np.mean(left_x))
    right_mean = float(np.mean(right_x))
    lane_width = float(right_mean - left_mean)
    curvature = float(np.std(xs))
    y_spread = float(np.std(ys))

    return [left_mean, right_mean, lane_width, curvature, y_spread]
