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
    if 0 < angle < 180:
        angle = angle**2 * 0.00123457 + 0.7777777 * angle
    if angle == 270:
        angle = 265
    return angle * math.pi / 180

# Locates the exact location of tip using yolo, and adds result to queue. 
# Ran concurrently with gcode parser.
# Return 0 if no tip is found
def yolov8_correct(q: queue.Queue, img_path: str, x: int, y: int, inference: Inference) -> None:
    img = cv2.imread(img_path)
    img, xadj = crop_around(img, x, y, 640, 640)
    box = inference.predict(img)
    if(box[0] == -1): q.put(0)
    else:
        xtip = int((box[0] + box[2])/2 + xadj)
        q.put(xtip-x)

def tip_tracker(
    g_path: str,
    fps: int,
    mTp: float,
    sX: int,
    sY: int,
    bed: Tuple[float, float, float],
    frame_path: str,
    accel: float,
) -> Tuple[List[int], List[float]]:    
    
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
    
    frame = 0
    correction_clock = 0
    thread_alarm = MIN_THREAD_ALARM
    total_correction = 0
    # Contains previous x offsets based off yolov8 correction
    spatial_offset_stack = []
    temporal_offset_stack = []
    # True - tracker is ahead, False - tracker is behind
    temporal_direction_stack = []

    inf = Inference(YOLO_PATH)
    q = queue.Queue()
    t1 = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
    t1.start() 
        
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
        time_for_move = how_long(abs(magnitude(position_vector)), abs(magnitude(curr_velocity_vector)), abs(final_speed), abs(max_speed), accel) * TIME_K
        
        frames_for_move = int(time_for_move * fps)
        if(frames_for_move != 0): 
            x_pixel_speed_per_frame = (position_vector[0] / frames_for_move) * mTp
            z_pixel_speed_per_frame = (position_vector[2] / frames_for_move) * mTp * 0.7
        final_pixel_position = [position_vector[0] * mTp + pixels[0], pixels[1] - position_vector[2] * mTp * 0.7]

        bed_angle = math.atan2(position_vector[1], position_vector[0]) * 180 / math.pi
        
        for i in range(frames_for_move):
            pixels = [pixels[0] + x_pixel_speed_per_frame, pixels[1] - z_pixel_speed_per_frame]
            pixel_locations.append(pixels.copy())
            angles.append(angle_adjustment(bed_angle))
            frame += 1
            correction_clock += 1
            if correction_clock == thread_alarm:
                t1.join()
                correction_clock = 0
                x_offset = q.get_nowait()
                spatial_offset_stack.append(x_offset)
                temporal_offset_stack.append(x_offset)
                t1 = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixel_locations[frame][0], pixel_locations[frame][1], inf))
                t1.start() 
                if spatial_offset_stack[-1] == 0: spatial_offset_stack.pop()
                # Spatial error
                if(len(spatial_offset_stack) == 2):
                    offset1, offset2 = spatial_offset_stack.pop(), spatial_offset_stack.pop()
                    if(offset1 < 0 and offset2 < 0 or offset1 > 0 and offset2 > 0):
                        correction = int((offset1 + offset2) / 2)
                        pixels[0] += correction
                        for tip_coords in pixel_locations[-thread_alarm*2:]:
                            tip_coords[0] += correction
                        if(thread_alarm >= 160): thread_alarm -= 10
                    else:
                        if(thread_alarm < 300): thread_alarm += 10
                # Temporal error
                #if(abs(correction1) < 5 and abs(correction2) >= 5 or abs(correction2) < 5 and abs(correction1) >= 5): 
                    #if(correction1 < correction2): accel *= 1.005
                    #else: accel *= 0.995
                    #thread_alarm = int(thread_alarm * 0.9)
                
        pixels = final_pixel_position
        angles.append(angle_adjustment(bed_angle))
        pixel_locations.append(pixels.copy())
        frame += 1
        correction_clock += 1
        if correction_clock == thread_alarm:
                t1.join()
                correction_clock = 0
                x_offset = q.get_nowait()
                spatial_offset_stack.append(x_offset)
                temporal_offset_stack.append(x_offset)
                t1 = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixel_locations[frame][0], pixel_locations[frame][1], inf))
                t1.start() 
                if spatial_offset_stack[-1] == 0: spatial_offset_stack.pop()
                # Spatial error
                if(len(spatial_offset_stack) == 2):
                    offset1, offset2 = spatial_offset_stack.pop(), spatial_offset_stack.pop()
                    if(offset1 < 0 and offset2 < 0 or offset1 > 0 and offset2 > 0):
                        correction = int((offset1 + offset2) / 2)
                        pixels[0] += correction
                        for tip_coords in pixel_locations[-thread_alarm*2:]:
                            tip_coords[0] += correction
                        if(thread_alarm >= 160): thread_alarm -= 10
                    else:
                        if(thread_alarm < 300): thread_alarm += 10
                # Temporal error
                #if(abs(correction1) < 5 and abs(correction2) >= 5 or abs(correction2) < 5 and abs(correction1) >= 5): 
                    #if(correction1 < correction2): accel *= 1.005
                    #else: accel *= 0.995
                    #thread_alarm = int(thread_alarm * 0.9)
        curr_bed_position = destination_bed_position.copy()
        curr_velocity_vector = final_velocity.copy()
        
    return pixel_locations, angles

def measure_diameter(video_path, g_code):
    cam = cv2.VideoCapture(video_path)
    
    currentframe = 0
    currentTime = 0 # Remember to divide time by 3
    g_line = 0
    
    bounding_boxes = tip_tracker(g_code, 30, 10.45212638, 657, 697, 82.554, 82.099, 1.8)
        
    while(True):
        ret,frame = cam.read()
        frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        draw.draw_rectangle("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/frame" + str(currentframe) + ".jpg", bounding_boxes[currentframe][0], bounding_boxes[currentframe][1], bounding_boxes[currentframe][2], bounding_boxes[currentframe][3])
        #time.sleep(3)
        print(bounding_boxes[currentframe])
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            
        
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
  
    cam.release()
    cv2.destroyAllWindows()

