import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_v8.YOLOv8 import YOLOv8

class Inference:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.3)

    def predict(self, img):
        boxes, scores, class_ids = self.yolov8_detector(img)
        if(len(boxes) == 0):
            boxes = [-1,-1,-1,-1]
        else:
            boxes = boxes[0]
        return [int(x) for x in boxes]

if __name__ == "__main__":
    inf = Inference("inference_/best.onnx")
    img = cv2.imread("/Users/brianprzezdziecki/Research/Mechatronics/STREAM Tip Detection.v1i.yolov7pytorch/train/images/frame0-2_jpg.rf.7b747ceadcc5769eb532d29d9d9eb200.jpg")
    print(img.shape)
    box = inf.predict(img)
    print(box)