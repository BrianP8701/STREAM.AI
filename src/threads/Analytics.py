'''
    Output is responsible for displaying the video and saving the video.
    
    Alongside displaying and saving the video, Output also runs MobileNet 
    to get the extrusion class and draws labels on the video.
'''
import src.threads.global_vars as GV
import src.helpers.helper_functions as helpers
import src.helpers.drawing_functions as d
import src.MobileNetv3.inference as Mobilenet
import src.helpers.preprocessing as preprocessing
import queue
import cv2

class Analytics:
    def __init__(self, display_video, display_fps, save_video, save_path, save_fps, resolution_percentage):
        self.display_video = display_video
        self.save_video = save_video
        self.save_path = save_path
        self.resolution_percentage = resolution_percentage
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.out = cv2.VideoWriter(save_path, fourcc, save_fps, (int(3840*(resolution_percentage/100)), int(2160*(resolution_percentage/100))))
        self.save_divisor = 30 / save_fps
        self.display_divisor = 30 / display_fps
    
    def start(self):
        helpers.print_text('Analytics thread started', 'blue')
        frame_index = 0
        while True:
            try:
                raw_frame = GV.video_queue.get(timeout=10)
            except queue.Empty:
                helpers.print_text('End of tracker', 'red')
                break
            
            frame = raw_frame.copy()
            frame = d.write_text_on_image(frame, f'Frame: {frame_index}', )
                
            if self.can_draw_box(frame_index):
                self.draw_tip_box(frame, frame_index)
                if len(GV.angles) > frame_index:
                    line = self.draw_line(frame, frame_index)
                    extrusion_box_coords = self.draw_extrusion_box(frame, frame_index, line)

                    if frame_index % self.display_divisor == 0 or frame_index % self.save_divisor == 0: # Only run inference when displaying or saving frame
                        extrusion_class = self.get_extrusion_class(extrusion_box_coords, raw_frame)
                        self.draw_extrusion_class(frame, extrusion_class)
                
            # Resize image for faster processing
            if self.save_video or self.display_video:
                frame = helpers.resize_image(frame, self.resolution_percentage)
            # Display video
            if self.display_video and frame_index % self.display_divisor == 0:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if frame_index == 1800:
                self.out.release()
                break
            if self.save_video and frame_index % self.save_divisor == 0:
                self.out.write(frame)
                
            frame_index += 1
            
        if self.save_video: self.out.release()
        cv2.destroyAllWindows
        
        helpers.print_text('Analytics thread done', 'red')
            
    def can_draw_box(self, frame_index):
        return (
            GV.tracking and 
            len(GV.screen_predictions) > frame_index and 
            GV.screen_predictions[frame_index][0] != -1
        )

    def draw_line(self, frame, frame_index):
        line = helpers.get_line(GV.screen_predictions[frame_index], GV.angles[frame_index]) 
        frame = d.draw_line(frame, line)
        return line
        
    def draw_tip_box(self, frame, frame_index):
        box = helpers.get_bounding_box(GV.screen_predictions[frame_index], 50)
        frame = d.draw_return(frame, round(box[0]), round(box[1]), round(box[2]), round(box[3]), thickness=3)
        return box
    
    def draw_extrusion_class(self, frame, extrusion_class):
        frame = d.write_text_on_image(frame, extrusion_class, position=(500, 300), font_scale=5, thickness=6)
        
    # Runs MobileNet to get extrusion class
    def get_extrusion_class(self, extrusion_box_coords, raw_frame):
        sub_img = helpers.crop_box_on_image(extrusion_box_coords, raw_frame)
        sub_img = preprocessing.gmms_preprocess_image(sub_img, 6)
        extrusion_class = Mobilenet.infer_image(sub_img, GV.mobile_model)
        return extrusion_class
        
    def draw_extrusion_box(self, frame, frame_index, line):
        box = helpers.crop_in_direction(GV.screen_predictions[frame_index], line)
        box = [round(box[0]), round(box[1]), round(box[2]), round(box[3])]
        frame = d.draw_return(frame, box[0], box[1], box[2], box[3], color=(0, 255, 0), thickness=3)
        return box