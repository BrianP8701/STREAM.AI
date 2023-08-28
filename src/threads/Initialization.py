'''
    Initialization.py contains the Initialization class, which is responsible for initializing the system.
    
    Initializes and synchronizes predictions with video using YOLO and signals.
'''
import src.threads.global_vars as GV
import src.helpers.helper_functions as helpers
import src.helpers.inference as inference
import src.threads.constants as c
import time
import gc
from pympler import asizeof

class Initialization:
    def __init__(self):
        self.initialization_video_buffer = []

    def start(self):
        self.initialize_ratio()

    def initialize_ratio(self):
        helpers.print_text('Initialization & Synchronization started', 'blue')
        leftmost_x = 9999999
        leftmost_x_signal_index = 0
        leftmost_img = None
        rightmost_x = -9999999
        rightmost_x_signal_index = 0
        rightmost_img = None
        banned_signals = []
        banned_signal_frames = []
        first_signal_index = 0
        second_signal_index = 0
        while True:
            # Loop until we find two signals far enough apart in the x direction
            break_here = False
            while True:
                for signal_frame in GV.initialization_frame_signal_buffer:
                    if signal_frame[0] in banned_signal_frames: continue
                    this_frame = signal_frame[1]
                    this_signal = signal_frame[2]
                    x = this_signal[0]
                    if x < leftmost_x: 
                        leftmost_x = x
                        leftmost_x_frame = signal_frame[0]
                        leftmost_x_signal_index = signal_frame[0]
                        leftmost_img = this_frame
                        first_signal_index = second_signal_index
                        second_signal_index = signal_frame[3]
                    if x > rightmost_x:
                        rightmost_x = x
                        rightmost_x_frame = signal_frame[0]
                        rightmost_x_signal_index = signal_frame[0]
                        rightmost_img = this_frame
                        first_signal_index = second_signal_index
                        second_signal_index = signal_frame[3]
                    if abs(rightmost_x - leftmost_x) > c.RATIO_INITIALIZATION_MIN_RANGE:
                        break_here = True
                if break_here: break
                time.sleep(0.1)

            # Get the bounding boxes of the two signals
            leftmost_box = inference.infer_large_image(leftmost_img, GV.yolo_model, 550)
            rightmost_box = inference.infer_large_image(rightmost_img, GV.yolo_model, 550)
            
            # If tip cannot be detected, find new signals
            if leftmost_box[0] == -1:
                leftmost_x = rightmost_x
                banned_signal_frames.append(leftmost_x_frame)
                continue
            if rightmost_box[0] == -1:
                rightmost_x = leftmost_x
                banned_signal_frames.append(rightmost_x_frame)
                continue
            
            # If the two signals are too far apart in the y direction, find new signals
            if leftmost_box[1] - rightmost_box[1] > c.ACCEPTABLE_Y_DIFFERENCE:
                rightmost_x = -9999999
                leftmost_x = 9999999
                banned_signals.append(leftmost_x_signal_index)
                banned_signals.append(rightmost_x_signal_index)
                continue
            
            break
        
        if leftmost_x_frame > rightmost_x_frame: 
            GV.yolo_history.append([rightmost_x_frame, helpers.get_center_of_box(rightmost_box)])
            GV.yolo_history.append([leftmost_x_frame, helpers.get_center_of_box(leftmost_box)])
        else: 
            GV.yolo_history.append([leftmost_x_frame, helpers.get_center_of_box(leftmost_box)])
            GV.yolo_history.append([rightmost_x_frame, helpers.get_center_of_box(rightmost_box)])

        GV.current_y = GV.yolo_history[0][1][1]
        # Calculate the ratio
        pixel_difference = abs(leftmost_box[0] - rightmost_box[0])
        millimeter_difference = rightmost_x - leftmost_x

        GV.ratio = pixel_difference / millimeter_difference
        del GV.initialization_video_buffer
        del GV.initialization_frame_signal_buffer
        gc.collect()
        self.initialize_screen_predictions(first_signal_index)

def initialize_screen_predictions(self, min_index):
    first_yolo = GV.yolo_history[0]
    first_yolo_frame_index = first_yolo[0]
    initial_first_yolo_prediction_index = GV.corner_indices[min_index]

    # Align predictions with first signal
    bed_predictions_reindex = first_yolo_frame_index - initial_first_yolo_prediction_index - 1 #initial_first_yolo_prediction_index
    GV.bed_predictions = helpers.modify_list(GV.bed_predictions, bed_predictions_reindex)
    GV.angles = helpers.modify_list(GV.angles, bed_predictions_reindex)
    GV.corner_indices = [corner_index + bed_predictions_reindex for corner_index in GV.corner_indices]
    
    # Fill the screen_predictions list with [-1, -1] for all frames before first yolo
    GV.screen_predictions = []
    
    for i in range(first_yolo_frame_index):
        GV.screen_predictions.append([-1, -1])
    
    tip = first_yolo[1].copy()
    GV.screen_predictions.append(tip.copy())
        
    for i in range(first_yolo_frame_index, len(GV.bed_predictions)):
        x_millimeter_change = GV.bed_predictions[i-1][0] - GV.bed_predictions[i][0]
        z_millimeter_change = GV.bed_predictions[i-1][2] - GV.bed_predictions[i][2]
                        
        tip[0] -= x_millimeter_change * GV.ratio
        tip[1] += z_millimeter_change * GV.ratio
        
        GV.screen_predictions.append([round(num) for num in tip.copy()])
    
    GV.tracking = True
    # Clear and free memory of frames (Was holding frames to be looked back at for this method)
    helpers.print_text('Initialization & Synchronization Done', 'green')
