'''
This file contains helper functions that are used in multiple places.
'''
import numpy as np
import cv2
import sys
import os
import json
import math

def parse_file(filename):
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


def crop_image_around_point(img, x, y, size):
    if isinstance(img, str):
        img = cv2.imread(img)

    # Calculate the coordinates of the crop
    left = int(x - size // 2)
    top = int(y - size // 2)
    right = int(x + size // 2)
    bottom = int(y + size // 2)

    # Crop the image using numpy slicing
    img_cropped = img[top:bottom, left:right]

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

def get_queue_memory_usage(q):
    temp_list = []
    total_size = sys.getsizeof(q)

    while not q.empty():
        item = q.get()
        total_size += sys.getsizeof(item)
        temp_list.append(item)

    for item in temp_list:
        q.put(item)

    return total_size

def print_text(text, text_color='white', bg_color='black'):
    text_colors = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }
    
    bg_colors = {
        'black': '40',
        'red': '41',
        'green': '42',
        'yellow': '43',
        'blue': '44',
        'magenta': '45',
        'cyan': '46',
        'white': '47'
    }
    
    print(f"\033[{text_colors[text_color]}m\033[{bg_colors[bg_color]}m{text}\033[0m")
        
def modify_list(lst, num):
    if num > 0:
        return [-1] * num + lst
    elif num < 0:
        return lst[abs(num):]
    else:
        return lst

def find_matching_index(bed_predictions, corner_indices, target_coords):
    """
    Find the index (from corner_indices) of a coordinate in bed_predictions 
    that matches the given target coordinate within an acceptable difference.

    Args:
    - bed_predictions (list of list): List of 3D coordinates, e.g., [[x1, y1, z1], [x2, y2, z2], ...].
    - corner_indices (list of int): List of indices referring to bed_predictions.
    - target_coords (list): Target 3D coordinate to match, e.g., [x, y, z].

    Returns:
    - int: Index from corner_indices of the matching coordinate or None if no match found.
    """
    
    acceptable_difference = 0.01
   
    rounded_bed_predictions = []
    for coord in bed_predictions:
        rounded_coord = [int(x * 100) / 100.0 for x in coord]
        rounded_bed_predictions.append(rounded_coord)
        
    for index in corner_indices:
        coord = rounded_bed_predictions[index]
        differences = [abs(coord[i] - target_coords[i]) for i in range(2)]
        if all(diff <= acceptable_difference for diff in differences):
            return index
    return None

def millis_to_frames(millis, fps):
    return round((millis / 1000) * fps)


def save_unique_image(folder_path, image_np):
    """ Save the given image to the specified folder with a unique name. """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Find a unique filename
    index = 0
    while True:
        filename = f"frame{index}.jpg"
        full_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(full_path):
            break
        index += 1
    
    # Save the image
    cv2.imwrite(full_path, image_np)

def save_list_to_json(data_list, destination_path):
    # Ensure the provided data is a list
    if not isinstance(data_list, list):
        raise ValueError("Provided data is not a list")

    # Create directories if they don't exist
    directory = os.path.dirname(destination_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the list to the JSON file
    with open(destination_path, 'w') as json_file:
        json.dump(data_list, json_file)


def resize_image(image, scale_percent):
    """
    Resize the image by the given scale percentage.
    
    :param image: Input image.
    :param scale_percent: Percentage by which the image should be scaled. E.g., 50 means the image will be half its original size.
    :return: Resized image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

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

def add_empty_frames_to_video(video_path, destination_path, delay_frames):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Couldn't open the video file.")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a video writer object
    out = cv2.VideoWriter(destination_path, fourcc, fps, (width, height))

    # Create an empty (black) frame
    empty_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add the empty frames to the beginning
    for _ in range(delay_frames):
        out.write(empty_frame)

    frame_index = delay_frames
    # Write the original video frames
    while True:
        ret, frame = cap.read()
        print(frame_index)
        frame_index += 1
        if not ret:
            break
        out.write(frame)

    # Release the video objects
    cap.release()
    out.release()
    
def remove_frames(input_path, output_path, num_frames_to_remove):
    """
    Removes a specified number of frames from the beginning of a video.
    
    Args:
    - input_path (str): Path to the input video file.
    - output_path (str): Path to save the output video file.
    - num_frames_to_remove (int): Number of frames to remove from the beginning of the video.

    Returns:
    None
    """

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to save the video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Video file ended

        frame_count += 1

        # If we've skipped enough frames, write the current frame to the output
        if frame_count > num_frames_to_remove:
            out.write(frame)

    # Release the VideoCapture and VideoWriter objects and close video file
    cap.release()
    out.release()
    
# Time travel. Every {interval} frames, either add or remove a frame based on the offset.
# Positive - add frames
# Negative - remove frames
def time_travel(screen_predictions, bed_predictions, angles, corner_indices,  current_frame_index, offset, interval=0):
    adjusted_screen_predictions = screen_predictions.copy()
    adjusted_bed_predictions = bed_predictions.copy()
    adjusted_angles = angles.copy()
    adjusted_corner_indices = corner_indices.copy()
    if offset > 0:  # We're ahead and need to add buffer frames.
        for i in range(offset):
            idx_to_modify = current_frame_index + i*interval
            if idx_to_modify < len(adjusted_screen_predictions):
                adjusted_screen_predictions.insert(idx_to_modify, adjusted_screen_predictions[idx_to_modify])
                adjusted_bed_predictions.insert(idx_to_modify, adjusted_bed_predictions[idx_to_modify])
                adjusted_angles.insert(idx_to_modify, adjusted_angles[idx_to_modify])
                first_signal_index = find_index(adjusted_corner_indices, idx_to_modify)
                for i in range(first_signal_index, len(adjusted_corner_indices)):
                    adjusted_corner_indices[i] += 1
            else:
                break  # Reached end of the list, can't add more

    elif offset < 0:  # We're behind and need to remove frames.
        for i in range(abs(offset)):
            idx_to_modify = current_frame_index + i*interval
            if idx_to_modify < len(adjusted_screen_predictions):
                del adjusted_screen_predictions[idx_to_modify]
                del adjusted_bed_predictions[idx_to_modify]
                del adjusted_angles[idx_to_modify]
                first_signal_index = find_index(adjusted_corner_indices, idx_to_modify)
                for i in range(first_signal_index, len(adjusted_corner_indices)):
                    adjusted_corner_indices[i] -= 1
            else:
                break  # Reached end of the list, can't remove more

    return adjusted_screen_predictions, adjusted_bed_predictions, adjusted_angles, adjusted_corner_indices


def least_squares_slope_stddev(x, y):
    '''
    Given a list of x and y values, uses least squares to calculate the slope and standard deviation of the slope
    '''
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


def compute_slope_from_range(coords: list, range_val: float) -> float:
    """
    Calculate the slope of the line segment defined by the most recent point and 
    the point which is 'range_val' units back in the x direction.
    
    Args:
    - coords: List of [x, y] sublists representing coordinates, sorted in ascending order by x.
    - range_val: Distance in the x direction to compute the slope.
    
    Returns:
    - Slope of the line segment.
    """
    
    # Get the most recent point (last in the list)
    x_recent, y_recent = coords[-1]
    
    # Find the point that is 'range_val' units back in x direction
    x_target = x_recent - range_val
    
    # Find the closest x value to x_target in the list (assuming the coordinates are sorted)
    target_coord = min(coords, key=lambda coord: abs(coord[0] - x_target))
    
    # Compute the slope
    slope = (y_recent - target_coord[1]) / (x_recent - target_coord[0])
    
    return slope


def find_index(lst, x):
    """Return the index of the first number in lst that is >= x."""
    for index, value in enumerate(lst):
        if value >= x:
            return index
    return None  # Return None if no such number exists in the list

def get_line(point, angle):
    """
    Returns the second point of line in the form of, of length 50, starting at (x, y) and at the given angle.
    
    0 degrees is to te right, 90 degrees is up.
    """
    angle_vector = [math.cos((angle*math.pi)/180)*50, math.sin((angle*math.pi)/180)*50] 
    angle_vector = [int(i) for i in angle_vector]
    return [round(point[0]), round(point[1]), round(point[0]+angle_vector[0]), round(point[1]-angle_vector[1])]

# Crops image in direction of angle. (For tracking the tip's extruded material)
def crop_in_direction(tip, line):
    vector = [line[2] - line[0], line[3] - line[1]]
    center_of_new_box = [tip[0] - vector[0] * 0.8, tip[1] - vector[1] * 0.8]
    box = get_bounding_box(center_of_new_box, 85)
    return box

# Makes some slight adjustments to calculated angle to account for parralax
def angle_adjustment(theta: float) -> float:
    angle = ((theta + 180) % 360)
    #if 0 < angle < 180:
        #angle = angle**2 * 0.00123457 + 0.7777777 * angle
    #if angle == 270:
        #angle = 265
    return angle

def crop_box_on_image(box, image):
    """
    Given a box represented as [x1, y1, x2, y2] and an image, returns the subsection of the image.

    Args:
    - box (list of int): [x1, y1, x2, y2] defining the top-left and bottom-right corners of the box.
    - image (numpy array): The image from which the subsection should be extracted.

    Returns:
    - numpy array: The extracted subsection of the image.
    """
    
    x1, y1, x2, y2 = box
    
    if len(image.shape) == 3:
        # For a color image
        return image[y1:y2, x1:x2, :]
    else:
        # For grayscale
        return image[y1:y2, x1:x2]
    