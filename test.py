from ultralytics import YOLO
 
def main():
    data = '../ultralytics/cfg/datasets/msrod-vh.yaml'
    model = YOLO('../runs/train/msod-vh/v8s-DGIoU/weights/best.pt')
    metrics = model.val(split='test', data=data, imgsz=1024, device=0, batch=16, workers=4, project='runs/test/msrod-vh', name='v8s-DGIoU')
    map75 = metrics.box.map75
    print(map75)

if __name__ == '__main__':
    main()
