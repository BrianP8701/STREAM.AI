import src.helpers.helper_functions as helpers
import src.MobileNetv3.inference as Mobilenet
import src.helpers.preprocessing as preprocessing
import src.threads.global_vars as GV

class Analytics:
    def __init__(self) :
        pass
    
    # Runs MobileNet to get extrusion class
    def get_extrusion_class(self, img):
        print(img.shape)
        extrusion_class = Mobilenet.infer_image(img, GV.mobile_model)
        return extrusion_class