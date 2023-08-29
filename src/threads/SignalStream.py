'''
    SignalStream is responsible for handling signals.
    
    Upon calling start(), SignalStream simulates the signal stream.
    The signal_router() routes the signals to the appropriate places as they stream in.
    The YOLO_thread() runs YOLO on corresponding frames when signals arrives and shares results with the spatial thread in ErrorCorrection.
'''
import time
import threading
import src.helpers.helper_functions as helpers
import src.threads.global_vars as GV
import src.helpers.inference as inference

class SignalStream:
    def __init__(self, signals_path, error_correction):
        self.signals_path = signals_path
        self.error_correction = error_correction
        self.signals_not_saved_to_initialization_buffer = [] # [[signal_index, time [x,y,z]]... ]

    def start(self):
        self._simulate_signal_stream()

    def _simulate_signal_stream(self):
        helpers.print_text('Signal stream started', 'blue')
        signal_list = helpers.parse_file(self.signals_path)
        GV.video_start_event.wait()
        threading.Thread(target=self._clear_initialization_video_buffer, args=()).start()        
    
        for signal in signal_list:
            self._ensure_correct_signal_speed(signal)
            self._signal_router(signal, GV.global_signal_index)  
            GV.signals.append([GV.global_signal_index, signal[0], signal[1]])   
            GV.global_signal_index += 1
        
        helpers.print_text('Signal stream ended', 'red')

    def _signal_router(self, signal, signal_index):
        signal_frame = helpers.millis_to_frames(signal[0], 30)
        if GV.tracking:
            threading.Thread(target=self.error_correction.temporal_thread, args=(signal_frame, signal_index,), daemon=True).start()
            threading.Thread(target=self.YOLO_thread, args=(signal_frame,), daemon=True).start()
        else:
            self._add_frame_to_initialization_buffer(signal, signal_index)

    # Everytime a signal arrives, run YOLO on the corresponding frame
    def YOLO_thread(self, signal_time_frame):
        if len(GV.screen_predictions) <= signal_time_frame: return
        # Make sure video buffer has the corresponding frame
        while True:
            buffer = GV.yolo_video_buffer.copy()
            buffer_start = buffer[0][0]
            buffer_end = buffer[-1][0]
            if signal_time_frame < buffer_start:
                helpers.print_text('Frame not found in buffer', 'blue')
                print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')
                raise Exception('Frame not found in buffer')
            if signal_time_frame > buffer_end:
                time.sleep(0.05)
            if signal_time_frame >= buffer_start and signal_time_frame < buffer_end:
                break
            
        # Find corresponding frame in video buffer
        for frame, image in buffer:
            if frame == signal_time_frame: 
                img = image.copy()
                break
        
        # Get real screen location
        predicted_screen_location = GV.screen_predictions[signal_time_frame]
        sub_img = helpers.crop_image_around_point(img, predicted_screen_location[0], predicted_screen_location[1], 640)
        try:
            real_screen_box = inference.yolo_inference(sub_img, GV.yolo_model)
        except:
            helpers.print_text(f"{sub_img}\n{predicted_screen_location}\n{signal_time_frame}\n{sub_img.shape}\n{type(sub_img)}\nYOLO failed\n", 'red')
            print(f'Frame {buffer[0][0]} to {buffer[-1][0]}\nSignal time frame: {signal_time_frame}\n')
        subimg_location = helpers.get_center_of_box(real_screen_box)
        real_screen_location = [subimg_location[0] - 320 + predicted_screen_location[0], subimg_location[1] - 320 + predicted_screen_location[1]]
        
        # Call spatial thread is YOLO prediction appears valid
        if not abs(real_screen_location[1] - GV.current_y) > 30: 
            threading.Thread(target=self.error_correction.spatial_thread, args=(signal_time_frame, real_screen_location,), daemon=True).start()
            
    def _ensure_correct_signal_speed(self, signal):
        signal_time = signal[0] / 1000.0  # Convert milliseconds to seconds
        time_to_wait = signal_time - (time.time() - GV.start_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

    # Add signals and corresponding images to initialization_frame_signal_buffer
    def _add_frame_to_initialization_buffer(self, signal, signal_index):
        self.signals_not_saved_to_initialization_buffer.append([signal_index, signal[0], signal[1]])
        for signal_ in self.signals_not_saved_to_initialization_buffer[:]:
            # Get index corresponding to signal_ in signals_not_saved_to_initialization_buffer
            corresponding_video_index = round((signal_[1] / 1000.0) * 30)
            try:
                index_in_initialization_buffer = corresponding_video_index - GV.initialization_video_buffer[0][0]
            except:
                return
            if len(GV.initialization_video_buffer) > index_in_initialization_buffer:
                GV.initialization_frame_signal_buffer.append([corresponding_video_index, signal_[0], 
                                                                GV.initialization_video_buffer[index_in_initialization_buffer][1]])
                # Delete all images from the initialization_video_buffer that are no longer needed.
                del self.signals_not_saved_to_initialization_buffer[0]
                del GV.initialization_video_buffer[0:index_in_initialization_buffer+1]
            else: break
    
    # Clear initialization_video_buffer when there are no signals arriving
    def _clear_initialization_video_buffer(self):
        was_previous_length_zero = False
        while True:
            if GV.tracking:
                break
            else:
                if was_previous_length_zero and len(self.signals_not_saved_to_initialization_buffer) == 0:
                    try: GV.initialization_video_buffer.clear()
                    except: pass
                    was_previous_length_zero = False
                    
                if len(self.signals_not_saved_to_initialization_buffer) == 0:
                    was_previous_length_zero = True
                time.sleep(10)