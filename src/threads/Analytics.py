import src.helpers.helper_functions as helpers
import src.MobileNetv3.inference as Mobilenet
import src.helpers.preprocessing as preprocessing
import src.threads.global_vars as GV

class Analytics:
    def __init__(self) :
        pass
    
    # Runs MobileNet to get extrusion class
    def get_extrusion_class(self, extrusion_box_coords, raw_frame):
        sub_img = helpers.crop_box_on_image(extrusion_box_coords, raw_frame)
        sub_img = preprocessing.gmms_preprocess_image(sub_img, 6)
        extrusion_class = Mobilenet.infer_image(sub_img, GV.mobile_model)
        return extrusion_class