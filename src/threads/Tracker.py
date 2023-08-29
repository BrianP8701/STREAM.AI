'''
    Tracker is the main class that handles the entire tracking process.
    It is responsible for starting all the threads.
'''
import threading
import src.helpers.helper_functions as helpers
import src.helpers.gcode_functions as g
import src.threads.constants as c
import src.threads.global_vars as GV
import src.helpers.drawing_functions as d
import src.helpers.preprocessing as preprocessing
import src.helpers.inference as inference
from src.YOLOv8.inference import Inference
import src.MobileNetv3.inference as Mobilenet
import cv2
import time
import queue
import gc
from src.threads.VideoStream import VideoStream
from src.threads.SignalStream import SignalStream
from src.threads.ErrorCorrection import ErrorCorrection
from src.threads.Initialization import Initialization
from src.threads.Analytics import Analytics
from src.threads.VideoOutput import VideoOutput

class Tracker:
    def __init__(self, video_path, gcode_path, signals_path, display_video=False, display_fps=6, save_video=False, save_fps=6, save_path=None, resolution_percentage=40):
        self.video_path = video_path
        self.gcode_path = gcode_path
        self.signals_path = signals_path
        
        self.Initialization_Handler = Initialization()
        self.Error_Correction = ErrorCorrection()
        self.Video_Stream = VideoStream(self.video_path)
        self.Signal_Stream = SignalStream(self.signals_path, self.Error_Correction)
        self.Video_Output = VideoOutput(display_video, display_fps, save_video, save_path, save_fps, resolution_percentage)

    def start(self):
        # Make initial predictions from gcode
        GV.bed_predictions, GV.angles, GV.corner_indices = g.gcode_parser(self.gcode_path, c.ACCELERATION, 30, c.TIME_K)
        
        threading.Thread(target=self.Video_Stream.start, args=()).start()
        threading.Thread(target=self.Signal_Stream.start, args=()).start()
        threading.Thread(target=self.Initialization_Handler.start, args=()).start()
        
        VideoOutput.start(self.Video_Output)