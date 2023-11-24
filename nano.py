from ultralytics import YOLO

#Loading the model
model = YOLO('yolov8n.pt')

#Training
results = model.train(
    data='pothole.yaml',
    imgsz=640,
    epochs=10,
    batch=8,
    name='yolov8n_custom'
)