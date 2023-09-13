'''
    MeasureMetrics is responsible for providing metrics driven research on the extrusion process.
    
    We will measure different performance critical aspects of the system, such as:

        - Speed
        - RAM usage
        - Classification accuracy
        - Diameter measurement accuracy
        
    Each of these metrics will be measured in a separate thread. Each thread will act as an observor.
'''
import src.variables.global_vars as GV
import src.helpers.helper_functions as helpers
import src.helpers.metrics as metrics
import src.helpers.drawing_functions as d
import src.helpers.preprocessing as preprocessing
from src.threads.Analytics import Analytics
import queue
import cv2
import threading
import psutil
import os
import time

class MeasureMetrics:
    def __init__(self, json_path, speed=False, ram=False, classification=False, diameter=False):
        self.json_path = json_path
        self.speed = speed
        self.ram = ram
        self.classification = classification
        self.diameter = diameter
        self.ram_history = [] # [time, ram_usage]
        self.speed_history = [] # [frame_index, time_since_last_inference]
        
    def start(self):
        if self.speed:
            threading.Thread(target=self.measure_speed, args=()).start()
        if self.ram:
            threading.Thread(target=self.measure_ram_usage, args=()).start()
        if self.classification:
            threading.Thread(target=self.measure_classification, args=()).start()
        if self.diameter:
            threading.Thread(target=self.measure_diameter, args=()).start()

    def measure_speed(self):
        '''
            Measure speed by measuring the time between MobileNet inferences
        '''
        last_frame_index = GV.measure_speed_queue.get() 
        while True:
            try:
                frame_index = GV.measure_speed_queue.get(timeout=5) 
            except queue.Empty:
                if GV.tracking_done:
                    break
                else:
                    continue
            frame_difference = frame_index - last_frame_index
            last_frame_index = frame_index
            print(f'Frame difference: {frame_difference}')
            self.speed_history.append([frame_index, frame_difference])
            
        metrics.update_json_dicts(self.json_path, {"speed": self.speed_history})
    
    def measure_ram_usage(self):
        '''
            Measure RAM USage
        '''
        while True:
            start_time = time.time()
            if GV.tracking_done:
                break
            ram_usage = self.measure_ram()
            self.ram_history.append([start_time, ram_usage])
            elapsed_time = time.time() - start_time
            if elapsed_time < 5:
                time.sleep(5 - elapsed_time)
                
        metrics.update_json_dicts(self.json_path, {"ram": self.ram_history})
    
    def measure_classification(self):
        pass
    
    def measure_diameter(self):
        pass
    
    def measure_ram(self):
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss
        return ram_usage