import queue
import src.helpers.helper_functions as helpers
import src.threads.global_vars as GV
import time
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
            img, img_with_gmms, extrusion_class = self.data_queue.get(timeout=10)
            self.process_data(img, img_with_gmms, extrusion_class)

    
    def process_data(self, img, img_with_gmms, extrusion_class):
        print(extrusion_class)
        print(img.shape)
        print(img_with_gmms.shape)
        # Write some code here
        pass