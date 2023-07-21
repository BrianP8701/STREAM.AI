import src.helper_functions as hf
import src.gcode_functions as g
import src.constants as c
from src.YOLOv8.inference import Inference
import src.inference as inference
import src.MobileNetv3.inference as mobilenet
import src.drawing_functions as d
import cv2
import threading
import time
import queue
import sys
import numpy as np

tracking = False
tracking_event = threading.Event()

ratio = 0
screen_predictions = []
acceleration = 64
# Contains any previous yolo predictions:
# [frame, [x1, y1, x2, y2]]
yolo_history = []
yolo_model = Inference(c.YOLO_PATH)
mobile_model = mobilenet.load_model(c.MOBILE_PATH)
# Begin video and signal streams
signal_queue = queue.Queue()
video_queue = queue.Queue()
# Contains the signals from the signal stream as:
# [time, [x, y, z]]...
signals = []
# Contains the frames from the video stream as numpy arrays
frames = []
# Contains the most recent 100 frames from the video stream as: [[frame, image], ...]
video_buffer = []
# Contains the bed predictions per frame from the gcode file as: [[x, y, z], ...]
bed_predictions = []
rounded_bed_predictions = []
# Contains the angles per frame, matching the bed predictions as: [angle, ...]
angles = []
# Indices of corners in bed predictions
corner_indices = []
# Current frame being streamed in right now
frame_index = 0
# [[frame, offset], ...]
spatial_offsets = []
temporal_offsets = []

def main_thread(video_path, gcode_path, signals_path, display_video=False, save_video=False, save_path=None):
    
    global frames
    global screen_predictions
    global tracking
    global bed_predictions
    global rounded_bed_predictions
    global angles
    global corner_indices
    
    bed_predictions, angles, corner_indices = g.gcode_parser(gcode_path, acceleration, 30)
    rounded_bed_predictions = [[round(num, 2) for num in sublist] for sublist in bed_predictions]

    threading.Thread(target=signal_thread, args=(signals_path,)).start()
    threading.Thread(target=video_thread, args=(video_path,)).start()
    threading.Thread(target=initialize_ratio, args=(frames, yolo_model), daemon=True).start()
    threading.Thread(target=signal_router, daemon=True).start()

    frame_index = 0
    first_frame = video_queue.get()
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (3840, 2160))
    
    while True:
        try:
            frame = video_queue.get(timeout=2)
        except queue.Empty:
            break

        if tracking:
            box = hf.get_bounding_box(screen_predictions[frame_index], 50)
            frame = d.draw_return(frame, round(box[0]), round(box[1]), round(box[2]), round(box[3]))
        
        if save_video and frame_index % 5 == 0:
            out.write(frame)
        
        if display_video and frame_index % 5 == 0:
            frame = hf.resize_image(frame, 40)
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
    global frame_index
    
    # Open video file
    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    
    start_time = time.time() # Get the current time
    target_time = start_time + 1/30 # The time we want to get the next frame
    frame_index = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        if not tracking: frames.append(frame)
        if len(video_buffer) < 200:
            video_buffer.append([frame_index, frame])
        else:
            video_buffer.pop(0)
            video_buffer.append([frame_index, frame])
        video_queue.put(frame)
        frame_index += 1
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
    
    signal_list = hf.parse_file(signals_path)
    
    start_time = time.time()  # Get the current time
    for signal in signal_list:
        signal_time = signal[0] / 1000.0  # Convert milliseconds to seconds
        time_to_wait = signal_time - (time.time() - start_time)
        if time_to_wait > 0:
            # Wait until the appropriate time has passed
            time.sleep(time_to_wait)
        # Add the signal to the queue
        if tracking: signal_queue.put(signal)
        signals.append(signal)
   
    hf.print_text('Signal stream ended', 'red')
    


# This thread receives signals and calls the appropriate functions
def signal_router():
    hf.print_text('Temporal thread activated', 'blue')
    global signals
    
    tracking_event.wait()
    time.sleep(0.5)
    
    while True:
        signal = signal_queue.get() 
        threading.Thread(target=temporal_thread, args=(signal,), daemon=True).start()
        time.sleep(0.1)
        buffer = video_buffer.copy()
        
        signal_time_frame = hf.millis_to_frames(signal[0], 30)
        signal_location = signal[1]
        print('Signal time frame: ' + str(signal_time_frame))
        # Find corresponding frame in video buffer
        for frame, image in buffer:
            if frame == signal_time_frame:  # Replace this with your actual condition
                img = image.copy()
                break
        predicted_screen_location = screen_predictions[signal_time_frame]
        sub_img = hf.crop_image_around_point(img, predicted_screen_location[0], predicted_screen_location[1], 640)
        real_screen_box = inference.yolo_inference(sub_img, yolo_model)
        real_screen_tip = hf.get_center_of_box(real_screen_box)
        real_screen_tip = [real_screen_tip[0] - 320 + predicted_screen_location[0], real_screen_tip[1] - 320 + predicted_screen_location[1]]
        print(real_screen_tip)
        print(frame)
        print()
        #spatial_thread(signal_time_frame, real_screen_tip)
        
        
def temporal_thread(signal):
    hf.print_text('Temporal thread activated', 'blue')
    
    signal_location = signal[1]
    for corner_index in corner_indices:
        if np.array_equal(np.array(rounded_bed_predictions[corner_index]), np.array(signal_location)):
            screen_time_frame = corner_index
            break
    signal_time_frame = hf.millis_to_frames(signal[0], 30)
    
    try:
        temporal_offset = signal_time_frame - screen_time_frame
        temporal_offsets.append([signal_time_frame, temporal_offset])
    except NameError:
        hf.print_text('Signal not found in bed predictions', 'red')
        return
    
    
    # Acceleration
    if len(temporal_offsets) > c.ACCELERATION_MIN_SIGNALS:
        slope, stdv = hf.least_squares_slope_stddev([pair[0] for pair in temporal_offsets], [pair[1] for pair in temporal_offsets])
        print('Slope: ' + str(slope))
        print('Stdv: ' + str(stdv))
    # Time Travel
    if len(temporal_offsets) > c.TIME_TRAVEL_MIN_SIGNALS:
        if max(temporal_offsets, key=lambda x: x[1])[1] - min(temporal_offsets, key=lambda x: x[1])[1] < c.TIME_TRAVEL_MAX_DEVIATION:
            hf.print_text('Time Travel', 'green')
            average_temporal_offset = sum(temporal_offsets) / len(temporal_offsets)
            print('Average temporal offset: ' + str(average_temporal_offset))
            screen_predictions, angles = hf.time_travel(screen_predictions, angles, signal_time_frame, average_temporal_offset)
            return
        
def spatial_thread(frame, real_screen_tip):
    hf.print_text('Spatial Thread Called', 'blue')
    
    spatial_error = [real_screen_tip[0] - screen_predictions[frame][0], real_screen_tip[1] - screen_predictions[frame][1]]

def ratio_thread():
    hf.print_text('Ratio Thread called', 'blue')
    signal_index = 0
    while True:
        
        signal_index += 1

def analytics_thread():
    hf.print_text('Analytics Thread Called', 'blue')
    time = 0
    while True:
        print(time)
        time += 1
        time.sleep(1)

def initialize_ratio(frames, yolo_model, min_range=5):
    hf.print_text('Initialization & Synchronization', 'blue')
    
    global yolo_history
    global ratio
    
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
        box1 = inference.infer_large_image(frames[start_frame], yolo_model, 550)
        box2 = inference.infer_large_image(frames[end_frame], yolo_model, 550)
        
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
        
        # If the two signals are too far apart in the y direction, find new signals
        if box1[1] - box2[1] > 50:
            max_x = -9999999
            min_x = 9999999
            banned_signals.append(min_index)
            banned_signals.append(max_index)
            continue

        
        if min_index < max_index:
            yolo_history.append([start_frame, hf.get_center_of_box(box1)])
            yolo_history.append([end_frame, hf.get_center_of_box(box2)])
            first_signal = signals[min_index]
        else:
            yolo_history.append([end_frame, hf.get_center_of_box(box2)])
            yolo_history.append([start_frame, hf.get_center_of_box(box1)])
            first_signal = signals[max_index]
            
        # Calculate the ratio
        pixel_difference = abs(box1[0] - box2[0])
        millimeter_difference = max_x - min_x

        ratio = pixel_difference / millimeter_difference
        
        threading.Thread(target=initialize_screen_predictions, args=(first_signal,), daemon=True).start()
        break
        
    
def initialize_screen_predictions(first_signal):
    global frames
    global screen_predictions
    global tracking
    global bed_predictions
    global angles
    global corner_indices
    global rounded_bed_predictions
    
    first_yolo = yolo_history[0]
    
    # Align bed predictions with first signal
    bed_prediction_index_matching_first_signal = rounded_bed_predictions.index(first_signal[1])
    bed_predictions_reindex = first_yolo[0] - bed_prediction_index_matching_first_signal - 1
    bed_predictions = hf.modify_list(bed_predictions, bed_predictions_reindex)
    rounded_bed_predictions = hf.modify_list(rounded_bed_predictions, bed_predictions_reindex)
    angles = hf.modify_list(angles, bed_predictions_reindex)
    
    corner_indices = [corner_index + bed_predictions_reindex for corner_index in corner_indices]
    
    # Fill the screen_predictions list with [-1, -1] for all frames before first yolo
    new_screen_predictions = []
    
    for i in range(first_yolo[0]):
        new_screen_predictions.append([-1, -1])
    
    tip = first_yolo[1].copy()
    new_screen_predictions.append(tip.copy())
    
    for i in range(first_yolo[0], len(bed_predictions)):
        x_millimeter_change = bed_predictions[i-1][0] - bed_predictions[i][0]
        z_millimeter_change = bed_predictions[i-1][2] - bed_predictions[i][2]
            
        tip[0] -= x_millimeter_change * ratio
        tip[1] -= z_millimeter_change * ratio
        
        new_screen_predictions.append([round(num) for num in tip.copy()])
        
    screen_predictions = new_screen_predictions
    tracking = True
    tracking_event.set()
    # Clear and free memory of frames (Was holding frames to be looked back at for this method)
    del frames
    import gc
    gc.collect()
    hf.print_text('Initialization & Synchronization Done', 'green')