import src.helper_functions as hf
import src.gcode_functions as g
import src.constants as c
import src.main_helper_functions as mh
from src.YOLOv8.inference import Inference
import src.MobileNetv3.inference as mobilenet
import src.drawing_functions as d
import cv2
import threading
import time
import queue

tracking = False
# Create a queue to hold frames for display
display_queue = queue.Queue()

def main_thread(video_path, gcode_path, signals_path):
    
    yolo_model = Inference(c.YOLO_PATH)
    mobile_model = mobilenet.load_model(c.MOBILE_PATH)
    # spatial_clock = 
    acceleration = 64
    bed_predictions, angles, corner_indices = g.gcode_parser(gcode_path, acceleration, 30)
    screen_predictions = []
    
    # Contains any previous yolo predictions:
    # [frame, [x1, y1, x2, y2]]
    yolo_history = []
    
    # Contains the frames from the video stream as numpy arrays
    frames = []
    
    # Contains the signals from the signal stream as:
    # [time, [x, y, z]]...
    signals = []
    
    # Begin video and signal streams
    signal_queue = queue.Queue()
    video_queue = queue.Queue()
    threading.Thread(target=signal_thread, args=(signals_path, signal_queue)).start()
    threading.Thread(target=video_thread, args=(video_path, video_queue)).start()
    
    # Begin tracker and temporal threads, listening for incoming frames and signals
    threading.Thread(target=tracker_thread, args=(video_queue, frames, screen_predictions), daemon=True).start()
    threading.Thread(target=temporal_thread, args=(signal_queue, signals), daemon=True).start()

    # Start the display thread in the main thread
    display_thread()
    
    # Initialization & Synchronization
    ratio = mh.initialize_ratio(signals, frames, yolo_model, yolo_history)
    first_yolo = yolo_history[0]
    screen_predictions = mh.initialize_screen_predictions(frames, first_yolo, bed_predictions, ratio)
    tracking = True
    
    print(type(screen_predictions))
    print(screen_predictions)

    

# This thread simulates the video stream
def video_thread(video_path, video_queue):
    print("\033[34mVideo stream beginning\033[0m")
    
    # Open video file
    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    
    start_time = time.time() # Get the current time
    target_time = start_time + 1/30 # The time we want to get the next frame
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # Make sure the video is playing at the correct speed
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        video_queue.put(frame)
        target_time += 1/30
        
    cam.release()
    print("\033[34mVideo stream ended\033[0m")
        
        
# This thread simulates the signal stream
def signal_thread(signals_path, signal_queue):
    print("\033[34mSignal stream beginning\033[0m") 

    signals = hf.parse_file(signals_path)
    
    start_time = time.time()  # Get the current time
    for signal in signals:
        signal_time = signal[0] / 1000.0  # Convert milliseconds to seconds
        time_to_wait = signal_time - (time.time() - start_time)
        if time_to_wait > 0:
            # Wait until the appropriate time has passed
            time.sleep(time_to_wait)
        
        # Add the signal to the queue
        signal_queue.put(signal)
   
    print("\033[34mSignal stream ended\033[0m")
 
# This thread provides the output video stream
def tracker_thread(video_queue, frames, screen_predictions):
    print("\033[34mTracker thread activated\033[0m")
    frame_index = 0
    while True:
        # Wait for a new frame to come in
        frame = video_queue.get() 
        
        frames.append(frame)

        if tracking:
            box = hf.crop_image_around_point(screen_predictions[frame_index])
            frame = d.draw_return(frame, box[0], box[1], box[2], box[3])
            
        # Add the frame to the display queue
        display_queue.put(frame)
        
        time.sleep(1/30)  # Process each frame for approximately 1/30th of a second
        frame_index += 1

def display_thread():
    while True:
        # Get the next frame from the display queue
        frame = display_queue.get()

        # Display the frame
        cv2.imshow('Real Time Image Stream', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    

        
        
def temporal_thread(signal_queue, signals):
    print("\033[34mTemporal thread activated\033[0m")
    while True:
        signal = signal_queue.get() 
        signals.append(signal)
        # Process the task here...

        
def spatial_thread():
    print("\033[34mthread started\033[0m")
    time = 0
    while True:
        print(time)
        time += 1
        time.sleep(1)
        
def ratio_thread():
    print("\033[34mthread started\033[0m")
    time = 0
    while True:
        print(time)
        time += 1
        time.sleep(1)
        
def analytics_thread():
    print("\033[34mthread started\033[0m")
    time = 0
    while True:
        print(time)
        time += 1
        time.sleep(1)
        



