'''
This file contains helper functions for the main thread.
'''
import src.inference as inference
import src.helper_functions as hf

def initialize_ratio(signals, frames, yolo_model, yolo_history, min_range=10):
    
    # We need to find two signals that are far enough apart so we can determine the ratio
    min_x = 9999999
    min_index = 0
    max_x = -9999999
    max_index = 0
    
    banned_signals = []
    
    while True:
        # Loop until we find two signals with a large enough range
        break_here = False
        while True:
            signals_length = len(signals)
            for i in range(signals_length):
                if i in banned_signals: continue
                x = signals[i][1][0]
                if x < min_x: 
                    min_x = x
                    min_index = i
                if x > max_x: 
                    max_x = x
                    max_index = i
                    
                if max_x - min_x > min_range:
                    break_here = True
                    
            if break_here: break

        # Get the frame numbers of the two signals
        start_frame = round((signals[min_index][0] / 1000.0) * 30)
        end_frame = round((signals[max_index][0] / 1000.0) * 30)
        
        # Get the bounding boxes of the two signals
        box1 = inference.infer_large_image(frames[start_frame], yolo_model, 600)
        box2 = inference.infer_large_image(frames[end_frame], yolo_model, 600)
        
        # If tip cannot be detected, find new signals
        if box1[0] == -1:
            min_x = max_x
            min_index = max_index
            banned_signals.append(min_index)
            continue
        if box2[0] == -1:
            max_x = min_x
            max_index = min_index
            banned_signals.append(max_index)
            continue
        
        if min_index > max_index:
            yolo_history.append([start_frame, hf.get_center_of_box(box1)])
            yolo_history.append([end_frame, hf.get_center_of_box(box2)])
        else:
            yolo_history.append([end_frame, hf.get_center_of_box(box2)])
            yolo_history.append([start_frame, hf.get_center_of_box(box1)])
            
        # Calculate the ratio
        pixel_difference = abs(box1[0] - box2[0])
        millimeter_difference = max_x - min_x
        millimeter_to_pixel_ratio = pixel_difference / millimeter_difference
        return millimeter_to_pixel_ratio
    
def initialize_screen_predictions(frames, first_yolo, bed_predictions, ratio):
    # Fill the screen_predictions list with [-1, -1] for all frames before first yolo
    screen_predictions = []
    for i in range(first_yolo[0]):
        screen_predictions.append([-1, -1])
    
    tip = first_yolo[1]
    screen_predictions.append(tip.copy())
    
    for i in range(len(bed_predictions) - first_yolo[0] + 1):
        x_millimeter_change = bed_predictions[i-1][0] - bed_predictions[i][0]
        z_millimeter_change = bed_predictions[i-1][2] - bed_predictions[i][2]
        
        tip[0] += x_millimeter_change * ratio
        tip[1] += z_millimeter_change * ratio

        screen_predictions.append([round(num) for num in tip.copy()])
    return screen_predictions
