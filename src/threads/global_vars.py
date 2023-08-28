# global_vars.py
import threading
import queue
from src.YOLOv8.inference import Inference
import src.MobileNetv3.inference as Mobilenet
import src.threads.constants as c

video_start_event = threading.Event() # Used to signal when video stream starts
video_queue = queue.Queue() # Used by tracking thread to save or display video

tracking = False # True if tracking, False if initializing
acceleration = c.ACCELERATION # Acceleration of the printer in mm/s^2
ratio = 0 # Ratio of pixels to mm
global_signal_index = 0 # Current signal index from signal stream
global_frame_index = 0 # Currect frame index from video stream
current_y = 0 # Current screen y position. If YOLO prediction deviates from this too much, it signals an error
start_time = 0 # Time when video starts

# Model Weights
yolo_model = Inference(c.YOLO_PATH)
mobile_model = Mobilenet.load_model(c.MOBILE_PATH)

yolo_history = [] # [frame, [x1, y1, x2, y2]]

signals = [] # [[signal_index, time, [x, y, z]]...

# During initialization video stream adds to this buffer
initialization_video_buffer = [] # [[frame_index, img]... ]
# During initialization, signal stream adds corresponding frames from initialization_video_buffer to this buffer and clears initialization_video_buffer
initialization_frame_signal_buffer = [] # [[frame_index, signal_index, img]... ]      
# Video stream adds to this buffer for YOLO inference during tracking
yolo_video_buffer = [] # [[frame_index, img]... ]

screen_predictions = [] # Contains the screen predictions per frame as: [[x, y], ...]

bed_predictions = [] # Contains the bed predictions per frame from the gcode file as: [[x, y, z], ...]
angles = [] # Contains the angles per frame, matching the bed predictions as: [angle, ...]
corner_indices = [] # Indices of corners in bed predictions

# [[frame, offset], ...]
x_spatial_offsets = []
y_spatial_offsets = []