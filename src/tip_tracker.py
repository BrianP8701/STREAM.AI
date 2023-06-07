"""
This module contains the tip tracking algorithm. 

I've organized the functions in this file in the following way:
    1. Helper Functions
    2. Error Detection
    3. Error Processing
    4. Error Correction
    5. Cropping
    6. Main Function
"""
import sys
import os
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)
import math
import numpy as np
import cv2
import threading
import queue
from .constants import *
from inference_.inference import Inference
from . import drawing as draw
from typing import List, Tuple

"""
    1. Helper Functions
"""
def normalize(vector):
    return vector / np.linalg.norm(vector)

def magnitude(vector):
    return np.linalg.norm(vector)

def quadratic(a, b, c):
    return (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)

# Returns the next line of gcode in the file and the index of the line after
def get_next_line(data: List[str], index):
    index += 1
    line_data = [-1.0, -1.0, -1.0, -1.0]
    while(index < len(data)):
        if(data[index][:2] == 'G1'): 
            line: list = data[index].split(" ")
            breakout = False
            for c in line:
                if(c[:1] == "X"): 
                    breakout = True
                    line_data[0] = float(c[1:])
                elif(c[:1] == "Y"): 
                    breakout = True
                    line_data[1] = float(c[1:])
                elif(c[:1] == "Z"): 
                    breakout = True
                    line_data[2] = float(c[1:])
                elif(c[:1] == "F"): line_data[3] = float(c[1:])
            if breakout: return line_data, index
        index += 1
    return line_data, -99

# Calculates how long it will take to finish move in seconds
def how_long(distance: float, curr_speed: float, final_speed: float, max_speed: float, acceleration: float) -> float:
    s1 = (max_speed**2 - curr_speed**2) / (2 * acceleration)
    if s1 >= distance:
        return (final_speed - curr_speed) / acceleration
    s2 = (max_speed**2 - final_speed**2) / (2 * acceleration)
    s3 = distance - s1 - s2
    return (max_speed - curr_speed) / acceleration + s3 / max_speed + (max_speed - final_speed) / acceleration

def update_this_line(line: List[float], curr_bed_position: np.ndarray, destination_bed_position: np.ndarray, 
                     position_vector: np.ndarray, next_max_speed: float) -> Tuple[float, float, np.ndarray]:
    for i in range(0,3):
        if(line[i] != -1): destination_bed_position[i] = line[i]
        else: destination_bed_position[i] = curr_bed_position[i]
    if(line[3] != -1): 
        max_speed = line[3] / 60
        if next_max_speed == 0: next_max_speed = max_speed
    position_vector = destination_bed_position - curr_bed_position
    max_velocity_vector = normalize(position_vector) * max_speed
    final_velocity = max_velocity_vector.copy()
    return max_speed, next_max_speed, final_velocity

def update_next_line(next_line: List[float], destination_bed_position: np.ndarray, next_destination_bed_position: np.ndarray, 
                     next_max_speed: np.ndarray) -> Tuple[float, np.ndarray]:
    for i in range(0,3):
        if(next_line[i] != -1): next_destination_bed_position[i] = next_line[i]
        else: next_destination_bed_position[i] = destination_bed_position[i]
    if(next_line[3] != -1): 
        next_max_speed = next_line[3] / 60
    next_position_vector = next_destination_bed_position - destination_bed_position
    next_max_velocity_vector = normalize(next_position_vector) * next_max_speed
    return next_max_speed, next_max_velocity_vector

def speed_at_end_of_this_move(final_velocity: np.ndarray, next_max_velocity_vector: np.ndarray):
    if final_velocity[0] >= 0 and next_max_velocity_vector[0] <= 0: final_velocity[0] = 0
    elif abs(final_velocity[0]) > abs(next_max_velocity_vector[0]): final_velocity[0] = next_max_velocity_vector[0]
    if final_velocity[1] >= 0 and next_max_velocity_vector[1] <= 0: final_velocity[1] = 0
    elif abs(final_velocity[1]) > abs(next_max_velocity_vector[1]): final_velocity[1] = next_max_velocity_vector[1]
    if final_velocity[2] >= 0 and next_max_velocity_vector[2] <= 0: final_velocity[2] = 0
    elif abs(final_velocity[2]) > abs(next_max_velocity_vector[2]): final_velocity[2] = next_max_velocity_vector[2]
    final_speed = magnitude(final_velocity)
    
def move_one_frame(x_pixel_speed_per_frame: float, z_pixel_speed_per_frame: float, pixel_locations: List[List], angles: List[float], bed_angle: float):
    pixels = [pixels[0] + x_pixel_speed_per_frame, pixels[1] - z_pixel_speed_per_frame]
    pixel_locations.append(pixels.copy())
    angles.append(angle_adjustment(bed_angle))
    return pixels
    
def move_last_frame(final_pixel_position: List[float], pixel_locations: List[List[float]], angles: List[float], bed_angle: float):
    pixels = final_pixel_position
    angles.append(angle_adjustment(bed_angle))
    pixel_locations.append(pixels.copy())
    return pixels
    
# Crops image around given point for given dimensions. Returns cropped image and x offset
def crop_around(img: np.ndarray, X: int, Y: int, length: int, wid: int) -> Tuple[np.ndarray, int]:
    if X - wid/2 < 0: 
        x = 0
        a = wid
    elif X + wid/2 > len(img[0]):
        x = len(img[0]) - wid - 1
        a = len(img[0]) - 1
    else:
        x = int(X - wid/2)
        a = int(X + wid/2)
    if Y - length/2 < 0:
        y = 0
        b = length
    elif Y + length/2 > len(img):
        y = len(img) - length - 1
        b = len(img) - 1
    else:
        y = int(Y - length/2)
        b = int(Y + length/2)
    return img[y:b, x:a], x

# Makes some slight adjustments to calculated angle to account for parralax
def angle_adjustment(theta: float) -> float:
    angle = ((theta + 180) % 360)
    #if 0 < angle < 180:
        #angle = angle**2 * 0.00123457 + 0.7777777 * angle
    #if angle == 270:
        #angle = 265
    return angle

"""
    2. Error Detection
"""
# Locates the exact location of tip using yolo, and adds result to queue. 
# Ran concurrently with gcode parser.
# Return 0 if no tip is found
def yolov8_correct(q: queue.Queue, img_path: str, x: int, y: int, inference: Inference) -> None:
    img = cv2.imread(img_path)
    img, xadj = crop_around(img, x, y, 640, 640)
    box = inference.predict(img)
    # Based off inference, add x_offset to queue
    if(box[0] == -1): q.put(0)
    else:
        xtip = int((box[0] + box[2])/2 + xadj)
        q.put(xtip-x)
        
# Gets temporal offset
def standard_horizontal_inference(frame: int, q: queue.Queue, frame_path: str, pixels: Tuple[int], 
                                  inf: Inference, x_pixel_speed_per_frame: float, bed_angle: float,
                                  temporal_offsets: List[float]):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, f'{frame_path}{frame}.jpg', pixels[0], pixels[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    spatial_offset = q.get_nowait()
    temporal_offset = spatial_offset / x_pixel_speed_per_frame
    if(spatial_offset == 0): return
   
    spatially_right = spatial_offset > 0
    # Ahead
    temporal_offset = abs(temporal_offset)
    # Behind
    if((not spatially_right and 240 >= bed_angle >= 120) or(spatially_right and 60 >= bed_angle >= 0)):
        temporal_offset = -1 * abs(temporal_offset)
    print(f"Standard Horizontal {frame} {temporal_offset}")
    
    if(len(temporal_offsets) >= 10):
        if(check_residual(temporal_offsets, [frame, temporal_offset], ACCEPTABLE_RESIDUAL)):
            temporal_offsets.append([frame, temporal_offset])
    else:
        temporal_offsets.append([frame, temporal_offset])

# Checks if we are temporally ahead or behind
def pre_vertical_inconclusive(frame1: int, frame2: int, q: queue.Queue, frame_path: str, 
                                  pixels1: Tuple[int], pixels2: Tuple[int], inf: Inference):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, f'{frame_path}{frame1}.jpg', pixels1[0], pixels1[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    spatial_offset1 = abs(q.get_nowait())
    
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, f'{frame_path}{frame2}.jpg', pixels2[0], pixels2[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    spatial_offset2 = abs(q.get_nowait())
    
    print(f'Spatial offset1: {spatial_offset1}         Spatial offset2: {spatial_offset2}')
    
    ahead = True
    
    if(spatial_offset1 > 10 and spatial_offset2 < 10): ahead = False
    elif(spatial_offset1 < 10 and spatial_offset2 > 10): ahead = True
    else: ahead = None
    return ahead

# Gets temporal offset
def pre_vertical_conclusive(frame: int, q: queue.Queue, frame_path: str, pixels: Tuple[Tuple[int]], 
                                  inf: Inference, temporal_offsets: List[float], temporal_error: bool, frames_behind: int):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, f'{frame_path}{frame}.jpg', pixels[0], pixels[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    initial_offset = abs(q.get_nowait())
    print(f'Prevertical conclusive {frame} {initial_offset}')
    if(initial_offset < ACCEPTABLE_PRE_VERTICAL_SPATIAL_ERROR): return frames_behind
    frame_adjust = frames_behind - 1
    if frame_adjust == -1: frame_adjust = 0
    # If we are behind, backtrack until no offset
    # If we are ahead go forward until no offset
    multiple = 1
    if(not temporal_error): multiple = -1
    current_frame = frame + multiple * frame_adjust
    
    ticker = 0
    # Loop until no offset
    while(True):
        if(ticker == 15): return frames_behind
        yolo_inference = threading.Thread(target=yolov8_correct, args=(q, f'{frame_path}{current_frame}.jpg', pixels[0], pixels[1], inf))
        yolo_inference.start() 
        yolo_inference.join()
        offset = abs(q.get_nowait())
        if offset < ACCEPTABLE_PRE_VERTICAL_SPATIAL_ERROR: break
        current_frame += multiple
        ticker += 1
        print(f"Pre-vertical loop. Offset: {offset}    Frame: {current_frame}")
    
    temporal_offset = current_frame - frame
    print(f'Temporal offset: {temporal_offset}')
    if(len(temporal_offsets) >= 10):
        if(check_residual(temporal_offsets, [frame, temporal_offset], ACCEPTABLE_RESIDUAL)):
           temporal_offsets.append([frame, temporal_offset])
        else: return frames_behind
    else: 
        temporal_offsets.append([frame, temporal_offset])
    
    return abs(current_frame - frame)
        
"""
    3. Error Processing
"""
def standard_horizontal(is_standard_horizontal: bool, standard_horizontal_clock: int, frame: int, q: queue.Queue, video_path: str, 
                        pixels: Tuple[int], frames_for_move: int, inf: Inference, x_pixel_speed_per_frame: float, bed_angle: float, 
                        temporal_offsets: List[float], temporal_errors: List[bool], i: int):
    if is_standard_horizontal and standard_horizontal_clock >= 60 and i == int(frames_for_move/2) and os.path.exists(f'{video_path}{frame}.jpg'):
        standard_horizontal_clock = 0
        standard_horizontal_inference(frame, q, video_path, pixels, inf, x_pixel_speed_per_frame, bed_angle, temporal_offsets)
                
        if(not is_conclusive and len(temporal_offsets) >= 1):
            temporal_errors.append(temporal_offsets[-1][1] > 0)
            print(temporal_errors)
            if(len(temporal_errors) >= 4):
                all_equal = temporal_errors[-4] == temporal_errors[-3] == temporal_errors[-2] == temporal_errors[-1]
                if all_equal:
                    is_conclusive = True
                    temporal_error = temporal_errors[-1]
        elif(len(temporal_offsets) >= 4):
            slope, stdv = least_squares_slope_stddev([pair[0] for pair in temporal_offsets], [pair[1] for pair in temporal_offsets])
        return standard_horizontal_clock, is_conclusive, temporal_error, slope, stdv
    return standard_horizontal_clock, is_conclusive, temporal_error, None, None

def pre_vertical(is_conclusive: bool, pre_vertical_clock: int, frame: int, q: queue.Queue, video_path: str, pixel_locations: Tuple[int],
                 final_pixel_position: Tuple[int], inf: Inference, temporal_offsets: List[float], temporal_error: bool, temporal_errors: List[bool],
                 frames_for_move: int,next_max_velocity_vector: np.ndarray, bed_angle: float, pixels: Tuple[int]) -> Tuple[int, bool, bool, float, float]:
    if(not is_conclusive and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0): 
        pre_vertical_clock = 0
        temporal_error = pre_vertical_inconclusive(frame, frame+frames_for_move+1, q, video_path, pixels, final_pixel_position, inf)
        if(temporal_error != None): 
            temporal_errors.append(temporal_error)
            if(len(temporal_errors) >= 4):
                all_equal = temporal_errors[-4] == temporal_errors[-3] == temporal_errors[-2] == temporal_errors[-1]
                if all_equal:
                    is_conclusive = True
                    temporal_error = temporal_errors[-1]
        return pre_vertical_clock, temporal_error, None, None, is_conclusive
                    
    elif(is_conclusive and temporal_error and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0):
        pre_vertical_clock = 0
        prev_temporal_offset = pre_vertical_conclusive(frame, q, video_path, final_pixel_position, inf, temporal_offsets, temporal_error, prev_temporal_offset)
        if(len(temporal_offsets) >= 4):
            slope, stdv = least_squares_slope_stddev([pair[0] for pair in temporal_offsets], [pair[1] for pair in temporal_offsets])
        return pre_vertical_clock, prev_temporal_offset, slope, stdv, is_conclusive
    elif(is_conclusive and not temporal_error and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0):
        pre_vertical_clock = 0
        prev_temporal_offset = pre_vertical_conclusive(frame, q, video_path, pixel_locations[frame-1], inf, temporal_offsets, temporal_error, prev_temporal_offset)
        if(len(temporal_offsets) >= 4):
            slope, stdv = least_squares_slope_stddev([pair[0] for pair in temporal_offsets], [pair[1] for pair in temporal_offsets])
        return pre_vertical_clock, prev_temporal_offset, slope, stdv, is_conclusive
    return pre_vertical_clock, prev_temporal_offset, None, None, is_conclusive
        
    
    
# Given a list of x and y values, uses least squares to calculate the slope and standard deviation of the slope
def least_squares_slope_stddev(x: List[float], y: List[float]):
    # Ensure the input arrays are NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate the means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (m) using the least squares method
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

    # Calculate the standard deviation of the slope
    n = len(x)
    residuals = y - (slope * x)
    stdev = np.sqrt(np.sum(residuals**2) / ((n - 2) * np.sum((x - x_mean)**2)))

    return slope, stdev

# Least squares to find best fit line
def best_fit_line(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Given a list of x, y values, remove anomalies based on residuals from best fit line
def remove_anomalies_line(data: List[List[float]], threshold=2.5) -> List[List[float]]:
    x = np.array([pair[0] for pair in data])
    y = np.array([pair[1] for pair in data])

    # Calculate the best-fit line
    slope, intercept = best_fit_line(x, y)

    # Calculate the residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Calculate the mean and standard deviation of the residuals
    mean_res, std_res = np.mean(residuals), np.std(residuals)

    # Calculate the Z-scores for the residuals
    z_scores = np.abs((residuals - mean_res) / std_res)

    # Filter the data using the Z-score threshold
    mask = z_scores < threshold
    x_filtered = x[mask]
    y_filtered = y[mask]

    return [list(pair) for pair in zip(x_filtered.tolist(), y_filtered.tolist())]

def check_residual(data, new_point, threshold):
    # Separate x and y values from the data
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    # Perform linear regression to find the best fit line
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate the residual for the new point
    expected_y = m * new_point[0] + c
    residual = abs(new_point[1] - expected_y)

    # Check if the residual is less than the threshold
    return residual < threshold

"""
    4. Error Correction
"""

"""
    5. Cropping
"""
# Crops image in direction of angle. (For tracking the tip's extruded material)
def crop_in_direction(tip: np.ndarray, theta: float) -> List[int]:
    box = [tip[0]-10, tip[1]-10, tip[0]+10, tip[1]+10]
    if(theta < 15 or theta > 345): box[2] = box[2] + 15
    elif(theta < 75):
        box[1] = box[1] - 15
        box[2] = box[2] + 20
    elif(theta < 105): box[1] = box[1] - 15
    elif(theta < 165):
        box[0] = box[0] - 15
        box[1] = box[1] - 15
    elif(theta < 195): box[0] = box[0] - 15
    elif(theta < 255):
        box[0] = box[0] - 15
        box[3] = box[3] + 15
    elif(theta < 285): box[3] = box[3] + 15
    elif(theta < 345):
        box[2] = box[2] + 15
        box[3] = box[3] + 15
    return box

"""
    6. Main Function
"""

# Variables that are a result of the fact that there is a lack of synchronization:
#   - sX, sY: The starting pixel location of the tip
#   - max_speed: The starting max speed
#   - next_max_speed: The next max speed after first line of gcode
#   - mTp: The conversion factor from mm to pixels
#   - bed: The starting bed position
#   - skip_lines: The number of lines to skip in the gcode file
# Upon synchronization, these variables will be removed. They will come from the gcode file, or be derived autonomously.

# Parses through gcode, and provides a prediction for the tip's location at each frame. 
# Built in error correction and error processing using YOLOv8.
def track(
    g_path: str,
    fps: int,
    mTp: float,
    sX: int,
    sY: int,
    bed: Tuple[float, float, float],
    video_path: str,
    time_k: float,
    skip_lines: int
) -> Tuple[list, list, list]:
       
# Open gcode file
    with open(g_path, 'r') as f_gcode:
        data = f_gcode.read()
        data: list = data.split("\n")
        
# Initialize tip tracker variables
    frame = 0
    pixels = [sX, sY + 10] # Current pixel location
    pixel_locations = [] # List of pixel locations
    angles = [] # List of angles
    g_index = skip_lines # Line in gcode
    curr_bed_position = np.array([bed[0], bed[1], bed[2]])
    destination_bed_position = np.array([0.0, 0.0, 0.0])
    next_destination_bed_position = np.array([0.0, 0.0, 0.0])
    position_vector = np.array([0.0,0.0,0.0])
    velocity_vector = np.array([0.0,0.0,0.0])
    # Max speeds are in mm/s
    max_speed = 8
    next_max_speed = 8
    temp, next_g_index = get_next_line(data, -1) # Next line in gcode
    pixel_locations.append(pixels)
    angles.append(0)

# Initialize error detection, processing and correction variables
    is_conclusive = False
    # True: Ahead, False: Behind
    temporal_error = None
    # List of booleans, telling us if we are ahead or behind. True: ahead, False: behind
    temporal_errors = []
    # List of temporal offsets 
    temporal_offsets = []
    # Clocks keep track of how much time has passed since last inference
    standard_horizontal_clock = 0
    pre_vertical_clock = 0
    # Temporal error measured from pre-vertical moves. 
    prev_temporal_offset = 0
    inf = Inference(YOLO_PATH)
    q = queue.Queue()
        
# Loop until end of gcode file
    while g_index != -99:
        line, g_index = get_next_line(data, g_index) # This frame's line of gcode
        next_line, next_g_index = get_next_line(data, next_g_index) # Next frame's line of gcode
        if not os.path.exists(f'{video_path}{frame}.jpg'): break

        # Update variables based on this line of gcode
        max_speed, next_max_speed, final_velocity = update_this_line(line, curr_bed_position, destination_bed_position, position_vector, next_max_speed)
        
        # Update variables based on next line of gcode
        next_max_speed, next_max_velocity_vector = update_next_line(next_line, destination_bed_position, next_destination_bed_position, next_max_speed)

        final_speed = speed_at_end_of_this_move(final_velocity, next_max_velocity_vector)
        time_for_move = how_long(abs(magnitude(position_vector)), abs(magnitude(velocity_vector)), abs(final_speed), abs(max_speed), ACCELERATION) * time_k
        frames_for_move = int(time_for_move * fps)
        x_pixel_speed_per_frame = ((position_vector[0] / time_for_move) * mTp) / fps
        z_pixel_speed_per_frame = ((position_vector[2] / time_for_move) * mTp * 0.7) / fps
        
        # Our prediction of pixel location and angle for this frame
        final_pixel_position = [position_vector[0] * mTp + pixels[0], pixels[1] - position_vector[2] * mTp * 0.7]
        bed_angle = math.atan2(position_vector[1], position_vector[0]) * 180 / math.pi

        #if(frames_for_move != 0): 
            #x_pixel_speed_per_frame = (position_vector[0] / frames_for_move) * mTp
            #z_pixel_speed_per_frame = (position_vector[2] / frames_for_move) * mTp * 0.7

# End of parser logic. We have determined our prediction of pixel location and angle for this frame.
# Now we will do error detection, processing and correction.

        is_standard_horizontal = False
        if((360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) and (frames_for_move >= 5) and max_speed < MAX_SPEED_FOR_STANDARD_HORIZONTAL and x_pixel_speed_per_frame > 0): is_standard_horizontal = True
        
        # We want to inference at start of parser move, and backtrack to see when the start of the real move was
        pre_vertical_clock, prev_temporal_offset, slope, stdv = pre_vertical(is_conclusive, pre_vertical_clock, frame, q, video_path, pixel_locations, final_pixel_position, inf, 
                                                                             temporal_offsets, temporal_error, temporal_errors, frames_for_move, next_max_velocity_vector, bed_angle, pixels)
        
        # Looping through frames for this line of gcode
        for i in range(frames_for_move):
            
            pixels = move_one_frame(x_pixel_speed_per_frame, z_pixel_speed_per_frame, pixel_locations, angles, bed_angle)
            frame += 1; pre_vertical_clock += 1; standard_horizontal_clock += 1
            if not os.path.exists(f'{video_path}{frame}.jpg'): break
            
            # Will perform standard horizontal inference if conditions are met
            standard_horizontal_clock, is_conclusive, temporal_error, slope, stdv = standard_horizontal(is_standard_horizontal, standard_horizontal_clock, frame, q, video_path, pixels, 
                                                                                                        frames_for_move, inf, x_pixel_speed_per_frame, bed_angle, temporal_offsets, temporal_errors, i)
        # Final variable updates, before moving on to next line of gcode
        pixels = move_last_frame(final_pixel_position, pixel_locations, angles, bed_angle)
        frame += 1; pre_vertical_clock += 1; standard_horizontal_clock += 1
        if not os.path.exists(f'{video_path}{frame}.jpg'): break
        curr_bed_position = destination_bed_position.copy()
        velocity_vector = final_velocity.copy()
        
        # We want to inference at end of parser move, and go forward to see when the start of the real move is
        pre_vertical_clock, prev_temporal_offset, slope, stdv = pre_vertical(is_conclusive, pre_vertical_clock, frame, q, video_path, pixel_locations, final_pixel_position, inf, 
                                                                             temporal_offsets, temporal_error, temporal_errors, frames_for_move, next_max_velocity_vector, bed_angle, pixels)
    return pixel_locations, angles, temporal_offsets

def measure_diameter(video_path, g_code):
    cam = cv2.VideoCapture(video_path)
    
    currentframe = 0
    currentTime = 0 # Remember to divide time by 3
    g_line = 0
    
    bounding_boxes = track(g_code, 30, 10.45212638, 657, 697, 82.554, 82.099, 1.8)
    
    while(True):
        ret,frame = cam.read()
        frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        draw.draw_rectangle(f"/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/frame{currentframe}.jpg", bounding_boxes[currentframe][0], bounding_boxes[currentframe][1], bounding_boxes[currentframe][2], bounding_boxes[currentframe][3])
        #time.sleep(3)
        print(bounding_boxes[currentframe])
        if ret:
            # if video is still left continue creating images
            name = f'./data/frame{currentframe}.jpg'
            
        
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
  
    cam.release()
    cv2.destroyAllWindows()

