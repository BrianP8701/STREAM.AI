import src.helper_functions as hf
import src.gcode_functions as g
import src.constants as c
from src.YOLOv8.inference import Inference
import src.inference as inference
import src.MobileNetv3.inference as mobilenet
import src.drawing_functions as d
import src.preprocessing as preprocessing
import cv2
import threading
import time
import queue
import numpy as np
import gc

video_start_event = threading.Event()
initiate_acceleration_thread = threading.Event()
initiate_recent_slope_queue = queue.Queue()
signal_queue = queue.Queue()
video_queue = queue.Queue()

tracking = False
acceleration = c.ACCELERATION
ratio = 0
screen_predictions = []
global_signal_index = 0
current_y = 0
start_time = 0

yolo_model = Inference(c.YOLO_PATH)
mobile_model = mobilenet.load_model(c.MOBILE_PATH)

yolo_history = [] # [frame, [x1, y1, x2, y2]]

signals = [] # [time, [x, y, z]]...
frames = [] # Contains the frames from the video stream as numpy arrays

video_buffer = [] # Contains the most recent 100 frames from the video stream as: [[frame, image], ...]

bed_predictions = [] # Contains the bed predictions per frame from the gcode file as: [[x, y, z], ...]
angles = [] # Contains the angles per frame, matching the bed predictions as: [angle, ...]
corner_indices = [] # Indices of corners in bed predictions

# Reference variables refer to the initial predictions
reference_bed_predictions = []
reference_angles = []
reference_corner_indices = []
reference_temporal_offsets = [] # [[frame, offset], ...], with respect to reference variables

# [[frame, offset], ...]
x_spatial_offsets = []
y_spatial_offsets = []

slopes = []
stdevs = []



def main_thread(video_path, gcode_path, signals_path, display_video=False, display_fps=6, save_video=False, save_fps=6, save_path=None, resolution_percentage=40):
    global frames
    global screen_predictions
    global tracking
    global bed_predictions
    global angles
    global corner_indices
    global initiate_recent_slope_queue
    global initiate_acceleration_thread
    
    bed_predictions, angles, corner_indices = g.gcode_parser(gcode_path, acceleration, 30)
    threading.Thread(target=video_thread, args=(video_path,)).start()
    threading.Thread(target=signal_thread, args=(signals_path,)).start()
    threading.Thread(target=signal_router, daemon=True).start()
    threading.Thread(target=initialize_ratio, args=(yolo_model,), daemon=True).start()
    threading.Thread(target=recent_slope_thread, daemon=True).start()
    threading.Thread(target=acceleration_thread, daemon=True).start()
    
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(save_path, fourcc, save_fps, (int(3840*(resolution_percentage/100)), int(2160*(resolution_percentage/100))))
    save_divisor = 30 / save_fps
    display_divisor = 30 / display_fps
        
    frame_index = 0
    while True:
        try:
            raw_frame = video_queue.get(timeout=10)
        except queue.Empty:
            hf.print_text('End of tracker', 'red')
            break
        
        frame = raw_frame.copy()
        frame = d.write_text_on_image(frame, f'Frame: {frame_index}', )
        
        if frame_index % c.RECENT_SLOPE_SAMPLE_INTERVAL == 0:
            initiate_recent_slope_queue.put(frame_index)
        
        if frame_index % c.ACCELERATION_SAMPLE_INTERVAL:
            initiate_acceleration_thread.set()
            
        if tracking and len(screen_predictions) > frame_index and screen_predictions[frame_index][0] != -1:
            box = hf.get_bounding_box(screen_predictions[frame_index], 50)
            frame = d.draw_return(frame, round(box[0]), round(box[1]), round(box[2]), round(box[3]), thickness=3)
            
            
        
        if tracking and len(screen_predictions) > frame_index and screen_predictions[frame_index][0] != -1 and len(angles) > frame_index:
            line = hf.get_line(screen_predictions[frame_index], angles[frame_index])
            frame = d.draw_line(frame, line)
            crop_box = hf.crop_in_direction(screen_predictions[frame_index], line)
            crop_box = [round(crop_box[0]), round(crop_box[1]), round(crop_box[2]), round(crop_box[3])]
            frame = d.draw_return(frame, crop_box[0], crop_box[1], crop_box[2], crop_box[3], color=(0, 255, 0), thickness=3)
            
            if frame_index % display_divisor == 0 or frame_index % save_divisor == 0:
                sub_img = hf.crop_box_on_image(crop_box, raw_frame)
                sub_img = preprocessing.gmms_preprocess_image(sub_img, 6)
           
                extrusion_class = mobilenet.infer_image(sub_img, mobile_model)
                frame = d.write_text_on_image(frame, extrusion_class, position=(500, 300), font_scale=5, thickness=6)
            
        if save_video or display_video:
            frame = hf.resize_image(frame, resolution_percentage)
            
        if save_video and frame_index % save_divisor == 0:
            out.write(frame)
        
        if display_video and frame_index % display_divisor == 0:
            cv2.imshow('Real Time Image Stream', frame)
            cv2.waitKey(1)
        
        frame_index += 1
    

    if save_video:
        out.release()
        
    cv2.destroyAllWindows()


# This thread simulates the video stream
def video_thread(video_path):
    hf.print_text('Video stream beginning', 'blue')
    global video_queue
    global frames
    global tracking
    global video_buffer
    global global_frame_index
    global start_time
    
    # Open video file
    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    
    start_time = time.time() # Get the current time
    video_start_event.set()
    target_time = start_time + 1/30 # The time we want to get the next frame
    global_frame_index = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        if not tracking: 
            try: frames.append(frame)
            except: pass
        if len(video_buffer) < 30:
            video_buffer.append([global_frame_index, frame])
        else:
            video_buffer.pop(0)
            video_buffer.append([global_frame_index, frame])
        video_queue.put(frame)
        global_frame_index += 1
        # Make sure the video is playing at the correct speed
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # queue_size = hf. get_queue_memory_usage(video_queue)
        # print("Queue size: " + str(queue_size) + " bytes")
        target_time += 1/30
        
    cam.release()
    hf.print_text('Video stream ended', 'red')
        
        
# This thread simulates the signal stream
def signal_thread(signals_path):
    hf.print_text('Signal stream beginning', 'blue')
    global signal_queue
    global signals
    global global_signal_index
    
    signal_list = hf.parse_file(signals_path)
    video_start_event.wait()
    for signal in signal_list:
        signal_time = signal[0] / 1000.0  # Convert milliseconds to seconds
        time_to_wait = signal_time - (time.time() - start_time)
        if time_to_wait > 0:
            # Wait until the appropriate time has passed
            time.sleep(time_to_wait)
        # Add the signal to the queue
        if tracking: signal_queue.put([signal, global_signal_index])
        signals.append(signal)
        global_signal_index += 1
    
    hf.print_text('Signal stream ended', 'red')


# This thread receives signals and calls the appropriate functions
def signal_router():
    hf.print_text('Signal Router activated', 'blue')
    global signals
    
    while True:
        signal, signal_index = signal_queue.get() 
        signal_time_frame = hf.millis_to_frames(signal[0], 30)
        threading.Thread(target=temporal_thread, args=(signal_time_frame, signal_index), daemon=True).start()
        threading.Thread(target=YOLO_thread, args=(signal_time_frame,), daemon=True).start()
        
# This thread runs YOLO and calls the appropriate functions
def YOLO_thread(signal_time_frame):
    if len(screen_predictions) <= signal_time_frame: return
    
    while True:
        buffer = video_buffer.copy()
        buffer_start = buffer[0][0]
        buffer_end = buffer[-1][0]
        if signal_time_frame < buffer_start:
            hf.print_text('Frame not found in buffer', 'blue')
            print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')
            raise Exception('Frame not found in buffer')
        if signal_time_frame > buffer_end:
            time.sleep(0.05)
        if signal_time_frame >= buffer_start and signal_time_frame < buffer_end:
            break
        
    # Find corresponding frame in video buffer
    found_frame = False
    for frame, image in buffer:
        if frame == signal_time_frame:  # Replace this with your actual condition
            img = image.copy()
            found_frame = True
            break
    
    predicted_screen_location = screen_predictions[signal_time_frame]
    sub_img = hf.crop_image_around_point(img, predicted_screen_location[0], predicted_screen_location[1], 640)

    try:
        real_screen_box = inference.yolo_inference(sub_img, yolo_model)
    except:
        hf.print_text(f"{sub_img}\n{predicted_screen_location}\n{signal_time_frame}\n{sub_img.shape}\n{type(sub_img)}\nYOLO failed\n", 'red')
        print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')


    subimg_location = hf.get_center_of_box(real_screen_box)
    real_screen_location = [subimg_location[0] - 320 + predicted_screen_location[0], subimg_location[1] - 320 + predicted_screen_location[1]]
    
    if not abs(real_screen_location[1] - current_y) > 30: 
        spatial_thread(signal_time_frame, real_screen_location)
    
    
def temporal_thread(frame, signal_index):    
    global slopes
    global stdevs
    global reference_temporal_offsets
    global screen_predictions
    global bed_predictions
    global temp_list
    global angles
    global corner_indices
    global recent_slope
    global last_recent_slope
    
    if signal_index < len(corner_indices): 
        reference_time_frame = reference_corner_indices[signal_index]
        screen_time_frame = corner_indices[signal_index]
    else: return
    
    reference_temporal_offset = frame - reference_time_frame        
    reference_temporal_offsets.append([frame, reference_temporal_offset])  
    screen_temporal_offset = frame - screen_time_frame
    screen_predictions, bed_predictions, angles, corner_indices = hf.time_travel(screen_predictions, bed_predictions, angles, corner_indices, frame, screen_temporal_offset)
    
    # print(f'Screen temporal offset: {screen_temporal_offset}')
    # print(f'Reference temporal offset: {reference_temporal_offset}')
    # print()
    # # Time Travel
    # if len(temporal_offsets) > c.TIME_TRAVEL_MIN_SIGNALS:
    #     # If the temporal offsets are within the max deviation, time travel
    #     if max(temporal_offsets, key=lambda x: x[1])[1] - min(temporal_offsets, key=lambda x: x[1])[1] < c.TIME_TRAVEL_MAX_DEVIATION:
    #         hf.print_text(f'Time Travel at {frame}', 'green')
    #         average_temporal_offset = round(sum(sublist[1] for sublist in temporal_offsets) / len(temporal_offsets))
    #         screen_predictions, angles, corner_indices = hf.time_travel(screen_predictions, angles, corner_indices, frame, average_temporal_offset)
    #         return
    

def spatial_thread(frame, real_screen_tip):   
    global screen_predictions
    global x_spatial_offsets
    global y_spatial_offsets
    global current_y
    
    spatial_offset = [real_screen_tip[0] - screen_predictions[frame][0], real_screen_tip[1] - screen_predictions[frame][1]]
    x_spatial_offsets.append([frame, spatial_offset[0]])
    y_spatial_offsets.append([frame, spatial_offset[1]])

    if len(x_spatial_offsets) > c.X_SPATIAL_MIN_SIGNALS: x_spatial_offsets.pop(0)
    if len(y_spatial_offsets) > c.Y_SPATIAL_MIN_SIGNALS: y_spatial_offsets.pop(0)
    
    # x spatial offsets
    if len(x_spatial_offsets) == c.X_SPATIAL_MIN_SIGNALS: # If we have enough signals
        if max(x_spatial_offsets, key=lambda x: x[1])[1] - min(x_spatial_offsets, key=lambda x: x[1])[1] < c.X_SPATIAL_MAX_DEVIATION: # If the spatial offsets are close enough together
            average_spatial_offset = sum(pair[1] for pair in x_spatial_offsets) / len(x_spatial_offsets)
            if abs(average_spatial_offset) > 5:
                hf.print_text('Adjust X spatially', 'green')
                for screeen_prediction in screen_predictions[frame:]:
                    screeen_prediction[0] += average_spatial_offset  
                x_spatial_offsets.clear()
    
    # y spatial offsets
    if len(y_spatial_offsets) == c.Y_SPATIAL_MIN_SIGNALS: # If we have enough signals
        if max(y_spatial_offsets, key=lambda x: x[1])[1] - min(y_spatial_offsets, key=lambda x: x[1])[1] < c.Y_SPATIAL_MAX_DEVIATION: # If the spatial offsets are close enough together
            average_spatial_offset = sum(pair[1] for pair in y_spatial_offsets) / len(y_spatial_offsets)
            if abs(average_spatial_offset) > 10: # If the spatial offset is large enough to be significant
                hf.print_text('Adjust Y spatially', 'green')
                for screeen_prediction in screen_predictions[frame:]:
                    screeen_prediction[1] += average_spatial_offset
                current_y += average_spatial_offset
                y_spatial_offsets.clear()


def acceleration_thread():
    while True:
        initiate_acceleration_thread.wait()
        initiate_acceleration_thread.clear()
        
        if len(reference_temporal_offsets) > c.ACCELERATION_MIN_SIGNALS:
            x_temporal_offsets, y_temporal_offsets = [pair[0] for pair in reference_temporal_offsets], [pair[1] for pair in reference_temporal_offsets]
            try:
                slope, stdv = hf.least_squares_slope_stddev(x_temporal_offsets, y_temporal_offsets)
                slopes.append([global_frame_index, slope])
                stdevs.append([global_frame_index, stdv])
            except: pass


def recent_slope_thread():
    while True:
        current_frame = initiate_recent_slope_queue.get()
        if len(reference_temporal_offsets) > c.ACCELERATION_MIN_SIGNALS and current_frame > c.RECENT_SLOPE_RANGE:
            recent_temporal_offsets = hf.filter_points_by_x_range(reference_temporal_offsets, current_frame - c.RECENT_SLOPE_RANGE, current_frame)
            x_temporal_offsets, y_temporal_offsets = [pair[0] for pair in recent_temporal_offsets], [pair[1] for pair in recent_temporal_offsets]
            
            # print(f'range is {current_frame - c.RECENT_SLOPE_RANGE} to {current_frame}')
            # change_in_slope, point_of_change = hf.piecewise_linear_regression(x_temporal_offsets, y_temporal_offsets, c.SLOPE_CHANGE_SENSITIVITY)
        
            # if change_in_slope:
            #     hf.print_text(f'Slope change detected at {current_frame}', 'green')
            # print()
                
                            

def ratio_thread():
    hf.print_text('Ratio Thread called', 'blue')
    signal_index = 0
        

def analytics_thread():
    hf.print_text('Analytics Thread Called', 'blue')
    time = 0
    while True:
        print(time)
        time += 1
        time.sleep(1)

def initialize_ratio(yolo_model):
    hf.print_text('Initialization & Synchronization started', 'blue')
    
    global yolo_history
    global ratio
    global current_y
    global frames
    
    # We need to find two signals that are far enough apart so we can determine the ratio
    leftmost_x = 9999999
    leftmost_x_signal_index = 0
    rightmost_x = -9999999
    rightmost_x_signal_index = 0
    banned_signals = []
    while True:
        # Loop until we find two signals with a large enough range
        break_here = False
        while True:
            signals_length = len(signals)
            for i in range(signals_length):
                if i in banned_signals: continue
                x = signals[i][1][0]
                if x < leftmost_x: 
                    leftmost_x = x
                    leftmost_x_signal_index = i
                if x > rightmost_x: 
                    rightmost_x = x
                    rightmost_x_signal_index = i
                    
                if abs(rightmost_x - leftmost_x) > c.RATIO_INITIALIZATION_MIN_RANGE:
                    break_here = True
            if break_here: break
            
        # Get the frame numbers of the two signals
        leftmost_x_frame = round((signals[leftmost_x_signal_index][0] / 1000.0) * 30)
        rightmost_x_frame = round((signals[rightmost_x_signal_index][0] / 1000.0) * 30)

        # Get the bounding boxes of the two signals
        leftmost_box = inference.infer_large_image(frames[leftmost_x_frame], yolo_model, 550)
        rightmost_box = inference.infer_large_image(frames[rightmost_x_frame], yolo_model, 550)
        
        # If tip cannot be detected, find new signals
        if leftmost_box[0] == -1:
            leftmost_x = rightmost_x
            leftmost_x_signal_index = rightmost_x_signal_index
            banned_signals.append(leftmost_x_signal_index)
            continue
        if rightmost_box[0] == -1:
            rightmost_x = leftmost_x
            rightmost_x_signal_index = leftmost_x_signal_index
            banned_signals.append(rightmost_x_signal_index)
            continue
        
        # If the two signals are too far apart in the y direction, find new signals
        if leftmost_box[1] - rightmost_box[1] > c.ACCEPTABLE_Y_DIFFERENCE:
            rightmost_x = -9999999
            leftmost_x = 9999999
            banned_signals.append(leftmost_x_signal_index)
            banned_signals.append(rightmost_x_signal_index)
            continue
        
        if leftmost_x_frame > rightmost_x_frame: 
            yolo_history.append([rightmost_x_frame, hf.get_center_of_box(rightmost_box)])
            yolo_history.append([leftmost_x_frame, hf.get_center_of_box(leftmost_box)])
        else: 
            yolo_history.append([leftmost_x_frame, hf.get_center_of_box(leftmost_box)])
            yolo_history.append([rightmost_x_frame, hf.get_center_of_box(rightmost_box)])

        current_y = yolo_history[0][1][1]
        # Calculate the ratio
        pixel_difference = abs(leftmost_box[0] - rightmost_box[0])
        millimeter_difference = rightmost_x - leftmost_x

        ratio = pixel_difference / millimeter_difference
        if leftmost_x_signal_index > rightmost_x_signal_index: min_index = rightmost_x_signal_index
        else: min_index = leftmost_x_signal_index
        del frames
        gc.collect()
        threading.Thread(target=initialize_screen_predictions, args=(min_index,), daemon=True).start()
        break
    
        
    
def initialize_screen_predictions(min_index):
    global screen_predictions
    global tracking
    global bed_predictions
    global angles
    global corner_indices
    global reference_bed_predictions
    global reference_angles
    global reference_corner_indices
    
    first_yolo = yolo_history[0]
    first_yolo_frame_index = first_yolo[0]
    initial_first_yolo_prediction_index = corner_indices[min_index]

    # Align predictions with first signal
    bed_predictions_reindex = first_yolo_frame_index - initial_first_yolo_prediction_index - 1 #initial_first_yolo_prediction_index
    bed_predictions = hf.modify_list(bed_predictions, bed_predictions_reindex)
    angles = hf.modify_list(angles, bed_predictions_reindex)
    corner_indices = [corner_index + bed_predictions_reindex for corner_index in corner_indices]
    
    # Fill the screen_predictions list with [-1, -1] for all frames before first yolo
    screen_predictions = []
    
    for i in range(first_yolo_frame_index):
        screen_predictions.append([-1, -1])
    
    tip = first_yolo[1].copy()
    screen_predictions.append(tip.copy())
        
    for i in range(first_yolo_frame_index, len(bed_predictions)):
        x_millimeter_change = bed_predictions[i-1][0] - bed_predictions[i][0]
        z_millimeter_change = bed_predictions[i-1][2] - bed_predictions[i][2]
                        
        tip[0] -= x_millimeter_change * ratio
        tip[1] += z_millimeter_change * ratio
        
        screen_predictions.append([round(num) for num in tip.copy()])
    
    reference_bed_predictions = bed_predictions.copy()
    reference_angles = angles.copy()
    reference_corner_indices = corner_indices.copy()
    
    tracking = True
    # Clear and free memory of frames (Was holding frames to be looked back at for this method)
    hf.print_text('Initialization & Synchronization Done', 'green')