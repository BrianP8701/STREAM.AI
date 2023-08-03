'''
This file contains functions that parse gcode files and return a list of bed locations,
and any other gcode related functions.
'''
import src.helper_functions as hf
import src.constants as c
import numpy as np
import math
import warnings
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

    # List of [x, y, z, speed]
    corners = simplify_gcode(gcode_path)
    frame = 0

    bed_predictions = []
    angle_list = []
    corner_indices = []
    velocity = np.array([0.0,0.0,0.0])
    leftover_frames = 0
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
        if final_speed > max_speed:
            final_speed = max_speed
            final_velocity = hf.normalize(displacement) * final_speed
        
        # In milliseconds
        time_for_move, achieved_final_speed = how_long(abs(distance), abs(speed), abs(final_speed), abs(max_speed), acceleration)
        time_for_move *= c.TIME_K
        achieved_final_velocity = hf.normalize(displacement) * achieved_final_speed
        
        frames_for_move = int(time_for_move * fps + leftover_frames)
            
        leftover_frames = (time_for_move * fps + leftover_frames) - frames_for_move
 
        if frames_for_move != 0: velocity = (displacement / frames_for_move)
        else: velocity = 0
        angle = math.atan2(displacement[1], displacement[0]) * 180 / math.pi
        
        for frame in range(frames_for_move):
            bed_predictions.append(location + velocity * frame)
            angle_list.append(angle)
            
        bed_predictions.append(next_location)
        angle_list.append(angle)
        
        velocity = achieved_final_velocity
        
    corner_indices.append(len(bed_predictions)-1)
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
    line, g_index = get_first_line(gcode)
    corners.append(line)
    position = line
    line, g_index = get_next_line(gcode, g_index)
    
    while g_index != -1:
        for i in range (0,3):
            if line[i] == -1: line[i] = position[i]
        
        if line[3] == -1: line[3] = speed
        else: 
            line[3] = line[3] / 60
            speed = line[3]
        
        position = line
        corners.append(line)
        line, g_index = get_next_line(gcode, g_index)
        
    return corners

# Get first line of gcode where we know x, y and z
def get_first_line(data: list[str]):
    line_data = [-1.0, -1.0, -1.0, -1.0]
    index = 0
    
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

# Returns the next line of gcode starting with G1
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
            elif line_data[2] != -1: return line_data, index+1
            
        index += 1
        
    # Return -1 if no more lines
    return line_data, -1


def how_long(distance: float, curr_speed: float, final_speed: float, 
                        max_speed: float, acceleration: float) -> tuple:
    """
    Calculates the minimum time required to cover a given distance starting at curr_speed, 
    not exceeding max_speed, and finishing at final_speed. Returns a tuple (time, achieved_final_speed).
    """
    
    # s1: Distance covered while moving from curr_speed to max_speed (accelerating if curr_speed is less than max_speed and decelerating if curr_speed is greater than max_speed):
    # s2: Distance covered while moving from max_speed to final_speed (accelerating if max_speed is less than final_speed and decelerating if max_speed is greater than final_speed):
    # s3: Distance covered while moving from curr_speed to final_speed:
    s1, t1, s2, t2, s3 = compute_travel_parameters(curr_speed, final_speed, max_speed, acceleration)
    accelerating = final_speed > curr_speed
    
    if s1 > distance: # We can't reach max speed
        if s3 > distance: # We can't reach final speed either
            warnings.warn("Failed to reach the desired final speed within the given distance. Optimal Travel Time function.")         
            # Calculate the final speed achievable within the given distance
            if accelerating: value_inside_sqrt = curr_speed**2 + 2*acceleration*distance
            else: value_inside_sqrt = curr_speed**2 - 2*acceleration*distance
            achievable_speed_change = value_inside_sqrt**0.5
            achieved_final_speed = curr_speed + achievable_speed_change
            time_to_achieve_speed_change = achievable_speed_change / acceleration
            return time_to_achieve_speed_change, achieved_final_speed
        
        else: # We can reach final speed. We want to accelerate as much as we can, then decelerate to final speed
            return optimal_travel_time(distance, curr_speed, final_speed, acceleration)
        
    else: # We can reach max speed
        if s1 + s2 < distance: # We can reach max speed and final speed
            time_in_between = (distance - s1 - s2) / max_speed
            time = t1 + time_in_between + t2
            return time, final_speed

        else: # If we achieve max speed, we can't reach final speed. We want to accelerate as much as we can, then decelerate to final speed
            return optimal_travel_time(distance, curr_speed, final_speed, acceleration)


def compute_travel_parameters(curr_speed: float, final_speed: float, 
                                 max_speed: float, acceleration: float) -> dict:
    '''
        All inputs are expected to be positive magnitudes.
    
        s1: Distance covered while moving from curr_speed to max_speed (accelerating if curr_speed is less than max_speed and decelerating if curr_speed is greater than max_speed):
        s2: Distance covered while moving from max_speed to final_speed (accelerating if max_speed is less than final_speed and decelerating if max_speed is greater than final_speed):
        s3: Distance covered while moving from curr_speed to final_speed:

        For s1 and s2, the sign of the time (whether it's positive or negative) will indicate the direction of the acceleration or deceleration. If the time is positive, it means we're accelerating; if negative, we're decelerating.
    '''
    # s1 and time to do s1
    s1 = (max_speed**2 - curr_speed**2) / (2 * acceleration)
    time_s1 = abs((max_speed - curr_speed) / acceleration)
    
    # s2 and time to do s2
    s2 = (max_speed**2 - final_speed**2) / (2 * acceleration)
    time_s2 = abs((max_speed - final_speed) / acceleration)
    
    # s3 and time to do s3 (which is also direct_decel_distance and time to do direct_decel_distance)
    s3 = (final_speed**2 - curr_speed**2) / (2 * acceleration)
    
    return s1, time_s1, s2, time_s2, s3


def optimal_travel_time(distance, curr_speed, final_speed, acceleration):
     # If curr_speed equals final_speed, we split the distance in half.
    if curr_speed == final_speed:
        s_accel = distance / 2
        v_intermediate = (2 * acceleration * s_accel + curr_speed**2)**0.5
    else:
        # Start with an initial guess
        s_accel = distance / 2
        s_decel = distance - s_accel
        # Adjust by 10%, 1% and 0.1% of the total distance until we're close enough to the desired final_speed
        # If curr_speed is less than final_speed, we accelerate for more than half the distance.
        # If curr_speed is greater than final_speed, we accelerate for less than half the distance.
        for i in range(30):
            if i < 10: percent_adjustment = 0.1
            elif i < 20: percent_adjustment = 0.01
            else: percent_adjustment = 0.001
            # While maintaining the correct total distance, adjust s_accel and s_decel until final_speed is achieved
            # Compute v_intermediate after accelerating over s_accel
            v_intermediate = (2 * acceleration * s_accel + curr_speed**2)**0.5
            # Compute final speed after decelerating over s_decel from v_intermediate
            under_root = v_intermediate**2 - 2 * acceleration * s_decel
            if under_root < 0: v_final_after_decel = 0
            else: v_final_after_decel = under_root**0.5
            # Adjust s_accel and s_decel
            if v_final_after_decel < final_speed:
                # Initially we should be 
                s_accel += percent_adjustment * distance  # Some small percentage of total distance
            else:
                s_accel -= percent_adjustment * distance  # Some small percentage of total distance
                if s_accel < 0: s_accel = 0
                
    # Now we compute times
    t1 = (-curr_speed + v_intermediate) / acceleration
    t2 = (v_intermediate - final_speed) / acceleration  # Since we're decelerating
    return t1 + t2, final_speed

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