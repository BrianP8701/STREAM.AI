import src.inference as inference
import src.drawing_functions as drawing
from src.YOLOv8.inference import Inference
import src.constants as c
from src.threads import main_thread
import src.helper_functions as hf


#          This block runs YOLO inference on a single image
# videos = c.TEST_INPUTS[0]
# img = hf.get_frame(videos[0], 412)
# inf = Inference(c.YOLO_PATH)

# box = inference.infer_large_image(img, inf, 640)
# print(box)
# drawing.draw(img, box[0], box[1], box[2], box[3])



#           This block runs main thread
videos = c.TEST_INPUTS[0]
main_thread(videos[0], videos[1], videos[2])