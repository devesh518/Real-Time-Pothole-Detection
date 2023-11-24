from ultralytics import YOLO

#Loading the model
model = YOLO('yolov8s.pt')

#Training
results = model.train(
    data = 'pothole.yaml',
    imgsz = 640,
    epochs = 50,
    batch = 8,
    name = 'yolov8_50epochs'
)