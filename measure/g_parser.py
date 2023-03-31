import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import math
import numpy as np
import cv2
import threading
import queue
from constants import *
from inference.inference import Inference
import random

def draw(image_path, x, y, a, b):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    return image

def custom_draw(image, x, y, a, b):
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    return image

def draw_rectangle(image_path, x, y, a, b):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    cv2.imshow('', image)
    cv2.waitKey(10000)
    
def tip_line(img, theta, tip):
    angle_vector = [math.cos(theta)*50, math.sin(theta)*50] 
    angle_vector = [int(i) for i in angle_vector]
    img = cv2.line(img, [int(tip[0]), int(tip[1])], [int(tip[0])+angle_vector[0], int(tip[1])-angle_vector[1]], (255, 0, 0), 3)
    return img

def quadratic(a, b, c):
    return (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)

def get_next_line(data, index):
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

def normalize(vector):
    return vector / np.linalg.norm(vector)

def magnitude(vector):
    return np.linalg.norm(vector)

def how_long(distance, curr_speed, final_speed, max_speed, acceleration):
    if (max_speed**2 - curr_speed**2)/(2*acceleration) >= distance:
        return (final_speed - curr_speed)/acceleration
    s1 = (max_speed**2 - curr_speed**2)/(2*acceleration)
    s2 = (max_speed**2 - final_speed**2)/(2*acceleration)
    s3 = distance - s1 - s2
    t1 = (max_speed - curr_speed)/acceleration
    t2 = (max_speed - final_speed)/acceleration
    t3 = s3/max_speed
    total_time = t1 + t2 + t3
    return total_time

def crop_in_direction(tip, theta):
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

def crop_around(img, X, Y, length, wid):
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

def yolov8_correct(q, img_path, x, y, inference: Inference):
    img = cv2.imread(img_path)
    img, xadj = crop_around(img, x, y, 640, 640)
    box = inference.predict(img)
    if(box[0] == -1): q.put(0)
    else:
        xtip = int((box[0] + box[2])/2 + xadj)
        q.put(xtip-x)

def tip_tracker(g_path, fps, mTp, sX, sY, bed, frame_path):
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
    inf = Inference(YOLO_PATH)
    q = queue.Queue()
    t = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
    t.start() 
        
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
        time_for_move = how_long(abs(magnitude(position_vector)), abs(magnitude(curr_velocity_vector)), abs(final_speed), abs(max_speed), ACCELERATION) * TIME_K
        
        frames_for_move = int(time_for_move * fps)
        x_pixel_speed_per_frame = (position_vector[0] / frames_for_move) * mTp
        final_pixel_position = [position_vector[0] * mTp + pixels[0], pixels[1] - position_vector[2] * mTp * 0.7]
        z_pixel_speed_per_frame = (position_vector[2] / frames_for_move) * mTp * 0.7
        
        bed_angle = math.atan2(position_vector[1], position_vector[0]) * 180 / math.pi
        
        for i in range(frames_for_move):
            pixels = [pixels[0] + x_pixel_speed_per_frame, pixels[1] - z_pixel_speed_per_frame]
            pixel_locations.append(pixels)
            frame += 1
            correction_clock += 1
            if correction_clock == 120:
                t.join()
                correction_clock = 0
                correction = q.get_nowait()
                for tip_coords in pixel_locations[-120:]:
                    tip_coords[0] += correction
                pixels = [pixels[0], pixels[1]]
                t = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
                t.start() 
                
            angles.append(angle_adjustment(bed_angle))
 
        pixels = final_pixel_position
        pixel_locations.append(pixels.copy())
        frame += 1
        correction_clock += 1
        if correction_clock == 120:
                t.join()
                correction_clock = 0
                correction = q.get_nowait()
                for tip_coords in pixel_locations[-120:]:
                    tip_coords[0] += correction
                pixels = [pixels[0] + correction, pixels[1]]
                t = threading.Thread(target=yolov8_correct, args=(q, frame_path + str(frame) + '.jpg', pixels[0], pixels[1], inf))
                t.start() 
        angles.append(angle_adjustment(bed_angle))
        
        curr_bed_position = destination_bed_position.copy()
        curr_velocity_vector = final_velocity.copy()
        
    return pixel_locations, angles

def angle_adjustment(theta):
    angle = ((theta + 180) % 360)
    if(angle > 0 and angle < 180): angle = angle**2 * 0.00123457 + 0.7777777 * angle
    if(angle == 270): angle = 265
    angle = angle * math.pi / 180
    return angle

def measure_diameter(video_path, g_code):
    cam = cv2.VideoCapture(video_path)
    
    currentframe = 0
    currentTime = 0 # Remember to divide time by 3
    g_line = 0
    
    bounding_boxes = tip_tracker(g_code, 30, 10.45212638, 657, 697, 82.554, 82.099, 1.8)
        
    while(True):
        ret,frame = cam.read()
        frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        draw_rectangle("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/frame" + str(currentframe) + ".jpg", bounding_boxes[currentframe][0], bounding_boxes[currentframe][1], bounding_boxes[currentframe][2], bounding_boxes[currentframe][3])
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

# Video 1
tips, angles = tip_tracker("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/gcode1.gcode", 30, 15.45212638, 425, 405, [82.554, 82.099, 1.8], '/Users/brianprzezdziecki/Research/Mechatronics/data1/frame')
# Video 2
#tips, angles = tip_tracker("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/gcode2.gcode", 30, 14.45, 1108, 370, [120.857,110, 1.8], '/Users/brianprzezdziecki/Research/Mechatronics/data2/frame')

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
video_filename = 'test_V3.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (1920, 1080))

frame = 0

while(frame < 7328):
    bounding_box = crop_in_direction(tips[frame], angles[frame]*180/math.pi)
    image = draw('/Users/brianprzezdziecki/Research/Mechatronics/data1/frame' + str(frame) + '.jpg', int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
    image = tip_line(image, angles[frame], tips[frame])
    cv2.putText(image, str(int(angles[frame]*180/math.pi)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(image)
    frame = frame + 1
    print(frame)

out.release()

# frame = 6764
# draw_rectangle('/Users/brianprzezdziecki/Research/Mechatronics/data/frame' + str(frame) + '.jpg', int(tips[frame][0]-5), int(tips[frame][1]-5), int(tips[frame][0]+5), int(tips[frame][1]+5))

print(len(tips))