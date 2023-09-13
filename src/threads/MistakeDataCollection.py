import queue
import src.helpers.helper_functions as helpers
import src.variables.global_vars as GV
import time
import src.helpers.drawing_functions as d
'''
    This class receives frames and their corresponding classes as 
    the system is running.
    
    The problem we are trying to address here is as follows:
    
        The MobileNet model has more than optimal capacity with the given dataset.
        However we need to use MobileNet because we don't have enough data to train a model from scratch. 
        We need to use transfer learning.
        
        Thus, we simply have no choice now but to collect a more diverse and representative dataset.
        
        The goal here is to see where the model makes mistakes in the system, and add those frames to the dataset.
'''

class MistakeDataCollection:
    def __init__(self):
        self.data_queue = GV.data_queue
    
    def data_stream(self):
        while not GV.tracking: # Wait for tracking to start
            time.sleep(1)
        while True:
            try:
                img, preprocessed_img, extrusion_class, frame_index = self.data_queue.get()
                self.process_data(img, preprocessed_img, extrusion_class, frame_index)
            except queue.Empty:
                helpers.print_text('Data is not streaming in', 'red')
                break
    
    def process_data(self, img, preprocessed_img, extrusion_class, frame_index):
                
        helpers.save_image(preprocessed_img, f'test_data/preprocessing/gmms/frame_{frame_index}.jpg')
        helpers.save_image(img, f'test_data/preprocessing/original/frame_{frame_index}.jpg')

        d.write_text_on_image(preprocessed_img, 'test', position=(5, 5), thickness=2, font_scale=0.2)
        d.write_text_on_image(preprocessed_img, f'{frame_index}', position=(5, 5), thickness=2, font_scale=0.2)
        d.write_text_on_image(img, f'{frame_index}', position=(5, 5), thickness=2, font_scale=0.2)
        d.write_text_on_image(img, f'{extrusion_class}', position=(5, 30), thickness=2, font_scale=0.2)

        helpers.save_image(preprocessed_img, 'g.jpg')
        helpers.save_image(img, 'kk.jpg')
        
        pass