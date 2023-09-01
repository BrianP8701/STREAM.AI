'''
    VideoStream is responsible for the video.
    
    Upon calling start(), VideoStream opens the video file and simulates the video stream.
    The frame_router() routes the frames to the appropriate places as they stream in.
'''
import time
import cv2
import src.helpers.helper_functions as helpers
import src.variables.global_vars as GV

class VideoStream:
    def __init__(self, video_path):
        self.video_path = video_path

    def start(self):
        helpers.print_text('Video stream beginning', 'blue')
        self._open_video_file()
        self._simulate_video_stream()

    def _open_video_file(self):
        self.cam = cv2.VideoCapture(self.video_path)
        GV.start_time = time.time()
        GV.video_start_event.set()
        self.target_time = GV.start_time + 1/30

    def _simulate_video_stream(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            self._frame_router(frame)
            GV.global_frame_index += 1
            self._ensure_correct_video_speed()
        self.cam.release()
        helpers.print_text('Video stream ended', 'red')

    def _frame_router(self, frame):
        if not GV.tracking: # save video frames for initialization
            try: GV.initialization_video_buffer.append([GV.global_frame_index,  frame])
            except: pass
        self._update_video_buffer(frame) # update video buffer for YOLO_thread
        GV.video_queue.put(frame) # send frame to tracking thread

    # Updates the video buffer used for YOLO_thread
    def _update_video_buffer(self, frame):
        if len(GV.yolo_video_buffer) < 30:
            GV.yolo_video_buffer.append([GV.global_frame_index, frame])
        else:
            GV.yolo_video_buffer.pop(0)
            GV.yolo_video_buffer.append([GV.global_frame_index, frame])

    def _ensure_correct_video_speed(self):
        sleep_time = self.target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.target_time += 1/30