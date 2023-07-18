'''
This file contains functions that parse gcode files and return a list of bed locations,
and any other gcode related functions.
'''
import numpy as np
import src.helper_functions as hf
import math

'''
Input: 
    gcode file
    acceleration value
    fps

Output:
    point_list: list of locations of tip for every frame
    angle_list: list of angles at each frame
                --> Frame by frame
                [a, b, c, d, e ...]
    corner_indices: list of indices in point_list that are corners
'''
def gcode_parser(gcode_path: str, acceleration: float, fps: int):

    # Open gcode file
    with open(gcode_path, 'r') as f_gcode:
        gcode = f_gcode.read()
        gcode: list = gcode.split("\n")

    # List of [x, y, z, speed] 
    corners = simplify_gcode(gcode_path)
    frame = 0    

    bed_predictions = []
    angle_list = []
    corner_indices = []
    velocity = np.array([0.0,0.0,0.0])
    total = 0
    
    # Loop through each corner
    for corner_index in range(len(corners)):
        # Skip if first or last line of gcode
        if corner_index == 0: continue
        if corner_index == len(corners) - 1: break
        
        if len(bed_predictions) != 0: corner_indices.append(len(bed_predictions)-1)
        else: corner_indices.append(0)
        
        # [x, y, z]
        location = np.array(corners[corner_index-1][0:3])
        next_location = np.array(corners[corner_index][0:3])
        next_next_location = np.array(corners[corner_index+1][0:3])
        # mm/s
        max_speed = corners[corner_index][3]
        next_max_speed = corners[corner_index+1][3]
        # vectors
        displacement = next_location - location
        next_displacement = next_next_location - next_location
        # scalars
        distance = hf.magnitude(displacement)
        speed = hf.magnitude(velocity)
        
        final_velocity = hf.normalize(displacement) * max_speed
        next_max_velocity = hf.normalize(next_displacement) * next_max_speed
        final_velocity, final_speed = speed_at_end_of_this_move(final_velocity, next_max_velocity)

        # In milliseconds
        time_for_move = how_long(abs(distance), abs(speed), abs(final_speed), abs(max_speed), acceleration) # * time_k
        frames_for_move = int(time_for_move * fps)
        velocity = (displacement / frames_for_move)
        angle = math.atan2(displacement[1], displacement[0]) * 180 / math.pi
        
        for frame in range(frames_for_move):
            bed_predictions.append(location + velocity * frame)
            angle_list.append(angle)
            
        bed_predictions.append(next_location)
        angle_list.append(angle)
        
        velocity = final_velocity
    
    return bed_predictions, angle_list, corner_indices


'''
Returns list of:
    [x, y, z, speed]
for each corner
'''
def simplify_gcode(gcode_path):
    corners = []
    
    # Open gcode file
    with open(gcode_path, 'r') as f_gcode:
        gcode = f_gcode.read()
        gcode: list = gcode.split("\n")
    
    speed = 0
    g_index = 0
    position = [-1.0,-1.0,-1.0, -1.0]
    
    while g_index != -1:
        line, g_index = get_next_line(gcode, g_index)
        for i in range (0,3):
            if line[i] == -1: line[i] = position[i]
        
        if line[3] == -1: line[3] = speed
        else: 
            line[3] = line[3] / 60
            speed = line[3]
        
        position = line
        corners.append(line)
        
    return corners


# Returns the next line of relevant gcode, and the next index to look at
def get_next_line(data: list[str], index):
                    # x, y, z, f
    line_data = [-1.0, -1.0, -1.0, -1.0]
    
    while(index < len(data)):
        
        # Break at end of file
        if(index >= len(data)): return line_data, -1
        
        if(data[index][:2] == 'G1'): 
            line: list = data[index].split(" ")
            
            for c in line:
                if(c[:1] == "X"): line_data[0] = float(c[1:])
                elif(c[:1] == "Y"): line_data[1] = float(c[1:])
                elif(c[:1] == "Z"): line_data[2] = float(c[1:])
                elif(c[:1] == "F"): line_data[3] = float(c[1:])

            # If all values are filled, return
            if all(i != -1 for i in line_data[:2]): return line_data, index+1
            
        index += 1
        
    # Return -1 if no more lines
    return line_data, -1


# Calculates how long it will take to finish move in seconds
def how_long(distance: float, curr_speed: float, final_speed: float, max_speed: float, acceleration: float) -> float:
    s1 = (max_speed**2 - curr_speed**2) / (2 * acceleration)
    if s1 >= distance:
        return (final_speed - curr_speed) / acceleration
    s2 = (max_speed**2 - final_speed**2) / (2 * acceleration)
    s3 = distance - s1 - s2
    return (max_speed - curr_speed) / acceleration + s3 / max_speed + (max_speed - final_speed) / acceleration


# Calculates the final speed of the move
def speed_at_end_of_this_move(final_velocity: np.ndarray,  next_max_velocity: np.ndarray) -> float:
    # Logic to tweak final velocity
    if final_velocity[0] >= 0 and next_max_velocity[0] <= 0: final_velocity[0] = 0
    elif abs(final_velocity[0]) > abs(next_max_velocity[0]): final_velocity[0] = next_max_velocity[0]
    if final_velocity[1] >= 0 and next_max_velocity[1] <= 0: final_velocity[1] = 0
    elif abs(final_velocity[1]) > abs(next_max_velocity[1]): final_velocity[1] = next_max_velocity[1]
    if final_velocity[2] >= 0 and next_max_velocity[2] <= 0: final_velocity[2] = 0
    elif abs(final_velocity[2]) > abs(next_max_velocity[2]): final_velocity[2] = next_max_velocity[2]
    final_speed = hf.magnitude(final_velocity)
    return final_velocity, final_speed