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
import numpy as np

video_start_event = threading.Event()
tracking = False
acceleration = c.ACCELERATION
ratio = 0
screen_predictions = []
# Contains the angles per frame, matching the bed predictions as: [angle, ...]
angles = []
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

# Indices of corners in bed predictions
corner_indices = []
# Current frame being streamed in right now
global_frame_index = 0
# [[frame, offset], ...]
x_spatial_offsets = []
y_spatial_offsets = []
temporal_offsets = []
signal_index = 0
current_y = 0
start_time = 0

temp_list = []

# TEMPORARY TRACKER VARIABLES
slopes = []
stdevs = []

def main_thread(video_path, gcode_path, signals_path, display_video=False, save_video=False, save_path=None):
    
    global frames
    global screen_predictions
    global tracking
    global bed_predictions
    global angles
    global corner_indices
    
    bed_predictions, angles, corner_indices = g.gcode_parser(gcode_path, acceleration, 30)
    threading.Thread(target=video_thread, args=(video_path,)).start()
    threading.Thread(target=signal_thread, args=(signals_path,)).start()
    threading.Thread(target=signal_router, daemon=True).start()
    threading.Thread(target=initialize_ratio, args=(frames, yolo_model), daemon=True).start()

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (3840, 2160))
        
    frame_index = 0
    while True:
        try:
            frame = video_queue.get(timeout=2)
        except queue.Empty:
            hf.print_text('End of tracker', 'red')
            break
        
        frame = frame.copy()
        frame = d.write_text_on_image(frame, f'Frame: {frame_index}', )
        
        if tracking:
            box = hf.get_bounding_box(screen_predictions[frame_index], 50)
            frame = d.draw_return(frame, round(box[0]), round(box[1]), round(box[2]), round(box[3]))
        
        if save_video and frame_index % 3 == 0:
            out.write(frame)
        
        if display_video and frame_index % 5 == 0:
            frame = hf.resize_image(frame, 40)
            cv2.imshow('Real Time Image Stream', frame)
            cv2.waitKey(1)

        if frame_index == 2000:
            print(slopes[-1][1])
            d.plot_points(slopes, 'Slopes0.jpg')
            d.plot_points(stdevs, 'Stdevs0.jpg')
            d.plot_points(temporal_offsets, 'TemporalOffsets0.jpg')
        if frame_index == 4000:
            print(slopes[-1][1])
            d.plot_points(slopes, 'Slopes1.jpg')
            d.plot_points(stdevs, 'Stdevs1.jpg')
            d.plot_points(temporal_offsets, 'TemporalOffsets1.jpg')
        if frame_index == 6000:
            print(slopes[-1][1])
            d.plot_points(slopes, 'Slopes2.jpg')
            d.plot_points(stdevs, 'Stdevs2.jpg')
            d.plot_points(temporal_offsets, 'TemporalOffsets2.jpg')
            hf.save_list_to_json(temp_list, 'temp_list.json')
        frame_index += 1

    d.plot_points(slopes, 'Slopes.jpg')
    d.plot_points(stdevs, 'Stdevs.jpg')
    
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
        if not tracking: frames.append(frame)
        if len(video_buffer) < 50:
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
    global signal_index
    
    signal_list = hf.parse_file(signals_path)
    video_start_event.wait()
    for signal in signal_list:
        signal_time = signal[0] / 1000.0  # Convert milliseconds to seconds
        time_to_wait = signal_time - (time.time() - start_time)
        if time_to_wait > 0:
            # Wait until the appropriate time has passed
            time.sleep(time_to_wait)
        # Add the signal to the queue
        if tracking: signal_queue.put(signal)
        signals.append(signal)
        signal_index += 1
    
    hf.print_text('Signal stream ended', 'red')


# This thread receives signals and calls the appropriate functions
def signal_router():
    hf.print_text('Signal Router activated', 'blue')
    global signals
    
    while True:
        signal = signal_queue.get() 
        signal_time_frame = hf.millis_to_frames(signal[0], 30)
        threading.Thread(target=temporal_thread, args=(signal_time_frame,), daemon=True).start()
        threading.Thread(target=YOLO_thread, args=(signal_time_frame,), daemon=True).start()
        
# This thread runs YOLO and calls the appropriate functions
def YOLO_thread(signal_time_frame):
    time.sleep(0.05)
    buffer = video_buffer.copy()
    
    # Find corresponding frame in video buffer
    found_frame = False
    for frame, image in buffer:
        if frame == signal_time_frame:  # Replace this with your actual condition
            img = image.copy()
            found_frame = True
            break
    if not found_frame: 
        hf.print_text('Frame not found in buffer', 'yellow')
        print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')
        raise Exception('Frame not found in buffer')
    
    predicted_screen_location = screen_predictions[signal_time_frame]
    sub_img = hf.crop_image_around_point(img, predicted_screen_location[0], predicted_screen_location[1], 640)

    try:
        real_screen_box = inference.yolo_inference(sub_img, yolo_model)
    except:
        print(predicted_screen_location[0])
        print(predicted_screen_location[1])
        hf.print_text(f"{sub_img}\n{predicted_screen_location}\n{signal_time_frame}\n{sub_img.shape}\n{type(sub_img)}\nYOLO failed\n", 'red')
        print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')


    subimg_location = hf.get_center_of_box(real_screen_box)
    real_screen_location = [subimg_location[0] - 320 + predicted_screen_location[0], subimg_location[1] - 320 + predicted_screen_location[1]]
    
    if not abs(real_screen_location[1] - current_y) > 30: 
        spatial_thread(signal_time_frame, real_screen_location)
    
    
def temporal_thread(frame):    
    global signal_index
    global slopes
    global stdevs
    global temporal_offsets
    global screen_predictions
    global temp_list
    global angles
    global corner_indices
    
    if signal_index < len(corner_indices): screen_time_frame = corner_indices[signal_index]
    else: return
    
    temporal_offset = frame - screen_time_frame
    
    temp_list.append([screen_time_frame, frame, temporal_offset])
    
    temporal_offsets.append([frame, temporal_offset])
    # Acceleration
    if len(temporal_offsets) > c.ACCELERATION_MIN_SIGNALS:
        slope, stdv = hf.least_squares_slope_stddev([pair[0] for pair in temporal_offsets], [pair[1] for pair in temporal_offsets])
        slopes.append([frame, slope])
        stdevs.append([frame, stdv])
        
        # if stdv < 1:
        #     print("Adjust Acceleration")
    # Time Travel
    if len(temporal_offsets) > c.TIME_TRAVEL_MIN_SIGNALS:
        # If the temporal offsets are within the max deviation, time travel
        if max(temporal_offsets, key=lambda x: x[1])[1] - min(temporal_offsets, key=lambda x: x[1])[1] < c.TIME_TRAVEL_MAX_DEVIATION:
            hf.print_text(f'Time Travel at {frame}', 'green')
            average_temporal_offset = round(sum(sublist[1] for sublist in temporal_offsets) / len(temporal_offsets))
            screen_predictions, angles, corner_indices = hf.time_travel(screen_predictions, angles, corner_indices, frame, average_temporal_offset)
            return
    

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

def initialize_ratio(frames, yolo_model, min_range=5):
    hf.print_text('Initialization & Synchronization started', 'blue')
    
    global yolo_history
    global ratio
    global current_y
    
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
        if box1[1] - box2[1] > 30:
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
        
        current_y = yolo_history[0][1][1]
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
    
    first_yolo = yolo_history[0]
    
    # Align predictions with first signal
    bed_predictions_reindex = first_yolo[0] - 1
    bed_predictions = hf.modify_list(bed_predictions, bed_predictions_reindex)
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
        tip[1] += z_millimeter_change * ratio
        
        new_screen_predictions.append([round(num) for num in tip.copy()])
        
    screen_predictions = new_screen_predictions
    tracking = True
    # Clear and free memory of frames (Was holding frames to be looked back at for this method)
    del frames
    import gc
    gc.collect()
    hf.print_text('Initialization & Synchronization Done', 'green')