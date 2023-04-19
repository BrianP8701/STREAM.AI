import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import math
import numpy as np
import cv2
import threading
import queue
from .constants import *
from inference_.inference import Inference
import random
from . import Drawing as draw
from typing import List, Tuple

def normalize(vector):
    return vector / np.linalg.norm(vector)

def magnitude(vector):
    return np.linalg.norm(vector)

def quadratic(a, b, c):
    return (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)

# Returns the next line of gcode in the file and the index of the line after
def get_next_line(data: List[str], index):
    index += 1
    line_data = [-1, -1, -1, -1]
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
    return line_data, -1

# Calculates how long it will take to finish move in seconds
def how_long(distance: float, curr_speed: float, final_speed: float, max_speed: float, acceleration: float) -> float:
    s1 = (max_speed**2 - curr_speed**2) / (2 * acceleration)
    if s1 >= distance:
        return (final_speed - curr_speed) / acceleration
    s2 = (max_speed**2 - final_speed**2) / (2 * acceleration)
    s3 = distance - s1 - s2
    return (max_speed - curr_speed) / acceleration + s3 / max_speed + (max_speed - final_speed) / acceleration

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
        
# Given a list of x and y values, uses least squares to calculate the slope and standard deviation of the slope
def least_squares_slope_stddev(data: List[List[float]]):
    # Ensure the input arrays are NumPy arrays
    x = np.array(np.array([pair[0] for pair in data]))
    y = np.array(np.array([pair[1] for pair in data]))

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

    return [x_filtered.tolist(), y_filtered.tolist()]

# Gets temporal offset
def standard_horizontal_inference(frame: int, q: queue.Queue, frame_path: str, pixels: Tuple[int], 
                                  inf: Inference, x_pixel_speed_per_frame: float, bed_angle: float,
                                  s_times: List[int], slopes: List[float], stdevs: List[float],
                                  temporal_offsets_1: List[float]):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    print(frame)
    spatial_offset = q.get_nowait()
    temporal_offset = spatial_offset / x_pixel_speed_per_frame
    if(spatial_offset == 0): return
    spatially_right = spatial_offset > 0
    # Ahead
    temporal_offset = abs(temporal_offset)
    # Behind
    if((not spatially_right and 240 >= bed_angle >= 120) or(spatially_right and 60 >= bed_angle >= 0)):
        temporal_offset = -1 * abs(temporal_offset)
                    
    temporal_offsets_1.append([frame, temporal_offset])
    if(len(temporal_offsets_1) >= 5):
        temporal_offsets_1 = remove_anomalies_line(temporal_offsets_1)
    if(len(temporal_offsets_1) >= 5):
        slope, stdev = least_squares_slope_stddev(temporal_offsets_1)
        s_times.append(frame)
        slopes.append(slope)
        stdevs.append(stdev)

# Checks if we are temporally ahead or behind
def pre_vertical_inconclusive(frame1: int, frame2: int, q: queue.Queue, frame_path: str, 
                                  pixels1: Tuple[int], pixels2: Tuple[int], inf: Inference):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame1) + '.jpg', pixels1[0], pixels1[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    print('{frame} prevertical inconclusive')
    spatial_offset1 = q.get_nowait()
    
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame2) + '.jpg', pixels2[0], pixels2[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    spatial_offset2 = q.get_nowait()
    
    ahead = True
    
    if(spatial_offset1 > 10 and spatial_offset2 > 10): print("Something is wrong")
    elif(spatial_offset1 < 10 and spatial_offset2 < 10): return
    elif(spatial_offset1 > 10 and spatial_offset2 < 10): ahead = False
    
    return ahead

# Gets temporal offset
def pre_vertical_conclusive(frame: int, q: queue.Queue, frame_path: str, pixels: Tuple[int], 
                                  inf: Inference, x_pixel_speed_per_frame: float, bed_angle: float,
                                  s_times: List[int], slopes: List[float], stdevs: List[float],
                                  temporal_offsets_2: List[float], temporal_error: bool):
    yolo_inference = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
    yolo_inference.start() 
    yolo_inference.join()
    print('{frame} prevertical conclusive')
    spatial_offset = q.get_nowait()
    if(spatial_offset < 5): return
    spatially_right = spatial_offset > 0
    
    # True: ahead, False: behind
    temporal_error = False
    # Ahead
    if((spatially_right and 240 >= bed_angle >= 120) or (not spatially_right and 60 >= bed_angle >= 0)):
        temporal_error = True  
    
    # ahead or behind, spatially right or left
    
    if(len(temporal_offsets_2) >= 5):
        temporal_offsets_2 = remove_anomalies_line(temporal_offsets_2)
    if(len(temporal_offsets_2) >= 5):
        slope, stdev = least_squares_slope_stddev(temporal_offsets_2)
        s_times.append(frame)
        slopes.append(slope)
        stdevs.append(stdev)


def tip_tracker(
    g_path: str,
    fps: int,
    mTp: float,
    sX: int,
    sY: int,
    bed: Tuple[float, float, float],
    frame_path: str,
    time_k: float,
) -> Tuple[list, list, list, list, list, list, list, list]:    
    
    with open(g_path, 'r') as f_gcode:
        data = f_gcode.read()
        data: list = data.split("\n")
        
    pixels = [sX, sY + 10]
    pixel_locations = []
    pixel_locations.append(pixels)
    angles = []
    angles.append(0)
    g_index = -1 # Line in gcode
    temp, next_g_index = get_next_line(data, -1)
    
    curr_bed_position = np.array([bed[0], bed[1], bed[2]])
    destination_bed_position = np.array([bed[0], bed[1], bed[2]])
    position_vector = np.array([0.0,0.0,0.0])
    curr_velocity_vector = np.array([0.0,0.0,0.0])
    next_destination_bed_position = np.array([bed[0], bed[1], bed[2]])
    max_speed = 0.0
    next_max_speed = 0.0
    
    temporary_coordinates = []
    temporary_coordinates.append([0,time_k])
    
    frame = 0
    standard_horizontal_clock = 0
    pre_vertical_clock = 0
    conclusive_state = False
    temporal_error = None

    # Temporal offsets for standard horizontal moves
    temporal_offsets_1 = []
    # Temporal offsets for pre-vertical moves
    temporal_offsets_2 = []
    # List of booleans, telling us if we are ahead or behind. True: ahead, False: behind
    temporal_errors = []
        
    # TO be deleted
    s_times = []
    slopes = []
    stdevs = []

    inf = Inference(YOLO_PATH)
    q = queue.Queue()
        
    # Main Loop
    while next_g_index != -1:
        line, g_index = get_next_line(data, g_index)
        next_line, next_g_index = get_next_line(data, next_g_index)

        for i in range(0,3):
            if(line[i] != -1): destination_bed_position[i] = line[i]
            else: destination_bed_position[i] = curr_bed_position[i]
        if(line[3] != -1): 
            max_speed = line[3] / 60
            if next_max_speed == 0: next_max_speed = max_speed
        
        if next_g_index != -1:
            for i in range(0,3):
                if(next_line[i] != -1): next_destination_bed_position[i] = next_line[i]
                else: next_destination_bed_position[i] = destination_bed_position[i]
            if(next_line[3] != -1): 
                next_max_speed = next_line[3] / 60
            next_position_vector = next_destination_bed_position - destination_bed_position
            next_max_velocity_vector = normalize(next_position_vector) * next_max_speed
        else:
            next_max_velocity_vector = np.array([0.0,0.0,0.0])

        position_vector = destination_bed_position - curr_bed_position
        max_velocity_vector = normalize(position_vector) * max_speed

        final_velocity = max_velocity_vector.copy()
        if final_velocity[0] >= 0 and next_max_velocity_vector[0] <= 0: final_velocity[0] = 0
        elif abs(final_velocity[0]) > abs(next_max_velocity_vector[0]): final_velocity[0] = next_max_velocity_vector[0]
        if final_velocity[1] >= 0 and next_max_velocity_vector[1] <= 0: final_velocity[1] = 0
        elif abs(final_velocity[1]) > abs(next_max_velocity_vector[1]): final_velocity[1] = next_max_velocity_vector[1]
        if final_velocity[2] >= 0 and next_max_velocity_vector[2] <= 0: final_velocity[2] = 0
        elif abs(final_velocity[2]) > abs(next_max_velocity_vector[2]): final_velocity[2] = next_max_velocity_vector[2]
        final_speed = magnitude(final_velocity)
        time_for_move = how_long(abs(magnitude(position_vector)), abs(magnitude(curr_velocity_vector)), abs(final_speed), abs(max_speed), ACCELERATION) * time_k
        
        frames_for_move = int(time_for_move * fps)
        if(frames_for_move != 0): 
            x_pixel_speed_per_frame = (position_vector[0] / frames_for_move) * mTp
            z_pixel_speed_per_frame = (position_vector[2] / frames_for_move) * mTp * 0.7
        final_pixel_position = [position_vector[0] * mTp + pixels[0], pixels[1] - position_vector[2] * mTp * 0.7]

        bed_angle = math.atan2(position_vector[1], position_vector[0]) * 180 / math.pi

        is_standard_horizontal = False
        if(360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120 and (frames_for_move >= 5) and max_speed < MAX_SPEED_FOR_STANDARD_HORIZONTAL): is_standard_horizontal = True
        # Inconclusive Pre-vertical
        if(not conclusive_state and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0): 
            pre_vertical_clock = 0
            temporal_errors.append(pre_vertical_inconclusive(frame, frame+frames_for_move+1, q, '{frame_path}{frame}.jpg', pixels, final_pixel_position, inf, x_pixel_speed_per_frame, bed_angle, s_times, slopes, stdevs, temporal_offsets_2))
            if(len(temporal_errors) > 3):
                all_equal = temporal_errors[-3] == temporal_errors[-2] == temporal_errors[-1]
                if all_equal:
                    conclusive_state = True
                    temporal_error = temporal_errors[-1]
            all_equal = temporal_errors[-3] == temporal_errors[-2] == temporal_errors[-1]

        # Conclusive pre-vertical if behind
        if(conclusive_state and not temporal_error and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0):
            pre_vertical_clock = 0
            pre_vertical_conclusive(frame, q, '{frame_path}{frame}.jpg', pixels, inf, x_pixel_speed_per_frame, bed_angle, s_times, slopes, stdevs, temporal_offsets_2, temporal_error)
            
        # Looping through frames for this line of gcode
        for i in range(frames_for_move):
            pixels = [pixels[0] + x_pixel_speed_per_frame, pixels[1] - z_pixel_speed_per_frame]
            pixel_locations.append(pixels.copy())
            angles.append(angle_adjustment(bed_angle))
            frame += 1
            pre_vertical_clock += 1
            standard_horizontal_clock += 1
            
            # Standard horizontal inference
            if is_standard_horizontal and standard_horizontal_clock >= 60 and i == int(frames_for_move/2) and os.path.exists('{frame_path}{frame}.jpg'):
                standard_horizontal_clock = 0
                standard_horizontal_inference(frame, q, '{frame_path}{frame}.jpg', pixels, inf, x_pixel_speed_per_frame, bed_angle, s_times, slopes, stdevs, temporal_offsets_1)
            
        pixels = final_pixel_position
        angles.append(angle_adjustment(bed_angle))
        pixel_locations.append(pixels.copy())
        frame += 1
        pre_vertical_clock += 1
        standard_horizontal_clock += 1
        # Conclusive pre-vertical if ahead
        if(conclusive_state and temporal_error and pre_vertical_clock > PRE_VERTICAL_CAP and (360 >= bed_angle >= 300 or 60 >= bed_angle >= 0 or 240 >= bed_angle >= 120) 
           and frames_for_move < 5 and next_max_velocity_vector[0] == 0 and abs(next_max_velocity_vector[1]) > 0):
            pre_vertical_clock = 0
            pre_vertical_conclusive(frame, q, '{frame_path}{frame}.jpg', pixels, inf, x_pixel_speed_per_frame, bed_angle, s_times, slopes, stdevs, temporal_offsets_2, temporal_error)
            
        curr_bed_position = destination_bed_position.copy()
        curr_velocity_vector = final_velocity.copy()
        
    return pixel_locations, angles, temporary_coordinates, temporal_offsets_1, temporal_offsets_2, slopes, stdevs, s_times

def measure_diameter(video_path, g_code):
    cam = cv2.VideoCapture(video_path)
    
    currentframe = 0
    currentTime = 0 # Remember to divide time by 3
    g_line = 0
    
    bounding_boxes = tip_tracker(g_code, 30, 10.45212638, 657, 697, 82.554, 82.099, 1.8)
        
    while(True):
        ret,frame = cam.read()
        frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        draw.draw_rectangle("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/frame{currentframe}.jpg", bounding_boxes[currentframe][0], bounding_boxes[currentframe][1], bounding_boxes[currentframe][2], bounding_boxes[currentframe][3])
        #time.sleep(3)
        print(bounding_boxes[currentframe])
        if ret:
            # if video is still left continue creating images
            name = './data/frame{currentframe}.jpg'
            
        
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
  
    cam.release()
    cv2.destroyAllWindows()

