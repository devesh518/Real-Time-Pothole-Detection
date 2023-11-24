from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO("best_50epochs.pt")

results = model.predict(source="demo1.mp4", show=True)
print(results)