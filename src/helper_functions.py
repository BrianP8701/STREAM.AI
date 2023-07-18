'''
This file contains helper functions that are used in multiple places.
'''
import numpy as np
from PIL import Image
import cv2
''' 
Parse a file with the format:
    2023-06-26 11:26:09
    Nozzle position X	Nozzle position Y	Nozzle position Z	
    4904	0.00	0.00	5.00	
    5518	5.00	0.00	5.00	
    6590	0.00	5.00	5.00	
    9193	0.00	0.00	2.00	
    ...

Returns a list of lists of the format:
    [[time, [x, y, z]], ...]
'''
def parse_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()[8:]  # Skip the first eight lines

    data = []
    for line in lines:
        parts = line.split()  # Split the line into parts
        time = int(parts[0])  # The first part is the time
        x = float(parts[1])  # The second part is the x-coordinate
        y = float(parts[2])  # The third part is the y-coordinate
        z = float(parts[3])  # The fourth part is the z-coordinate
        data.append([time, [x, y, z]])

    return data


def crop_image_around_point(img_path, x, y, size):
    # Open the image file
    img = Image.open(img_path)

    # Calculate the coordinates of the crop
    left = x - size // 2
    top = y - size // 2
    right = x + size // 2
    bottom = y + size // 2

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))
    
    return img_cropped

# Given a video and frame, return that frame as a numpy array
def get_frame(video_path, frame_number):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Set the current frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = video.read()

    # If the frame was read successfully, return it
    if ret:
        return frame
    else:
        return None

def get_center_of_box(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return [center_x, center_y]

def get_bounding_box(center, size):
    x, y = center
    half_size = size / 2
    return [x - half_size, y - half_size, x + half_size, y + half_size]

def read_image(img_path):
    img = cv2.imread(img_path)
    return img

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    
def save_image(img, img_path):
    # Save the image
    cv2.imwrite(img_path, img)

def magnitude(vector):
    return np.linalg.norm(vector)

def normalize(vector):
    return vector / np.linalg.norm(vector)