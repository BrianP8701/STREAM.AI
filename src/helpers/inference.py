'''
This file contains functions that perform object detection or classification.
'''
from src.YOLOv8.inference import Inference
import src.MobileNetv3.inference as Mobilenet
import cv2

# inf = Inference(c.YOLO_PATH)
def yolo_inference(img, model):
    if isinstance(img, str):
        img = cv2.imread(img)
        
    box = model.predict(img)
    return box


# model = mobilenet.load_model('src2/MobileNetv3/mob_l_gmms2_finetune.pt')
def mobile_inference(img, model):
    image_class = Mobilenet.infer_image(img, model)
    
    if image_class == 0: image_class = 'normal'
    elif image_class == 1: image_class = 'over'
    elif image_class == 2: image_class = 'under'
    return image_class


# YOLO inference on image larger than 640x640
def infer_large_image(img, model, stride=320):
    if isinstance(img, str):
        img = cv2.imread(img)
        
    # Initialize the bounding box to [-1, -1, -1, -1]
    bbox = [-1, -1, -1, -1]

    # Get the dimensions of the image
    height, width, _ = img.shape

    # Slide a 640x640 window across the image
    for y in range(0, height - 640, stride):
        for x in range(0, width - 640, stride):
            # Extract the window
            window = img[y:y+640, x:x+640]

            # Apply the yolo_inference method to the window
            window_bbox = yolo_inference(window, model)

            # If an object was detected in the window, update the bounding box
            if window_bbox != [-1, -1, -1, -1]:
                # Adjust the coordinates of the bounding box to be relative to the whole image
                bbox = [x + window_bbox[0], y + window_bbox[1], x + window_bbox[2], y + window_bbox[3]]

    return bbox

