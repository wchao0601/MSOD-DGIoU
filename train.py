
from ultralytics import YOLO


data='../ultralytics/cfg/datasets/msrod-vh.yaml'
# model = YOLOv10('/opt/data/private/CODE/yolov10/yolov10s.pt')
model = YOLO('yolov8s.pt')
project = 'runs/train'
name = 'msod-vh/v8s-DGIoU'
model.train(data=data, epochs=100, batch=16, imgsz=1024, name=name, device=0, project=project)