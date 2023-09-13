import src.helpers.helper_functions as helpers
import src.MobileNetv3.inference as Mobilenet
import src.helpers.preprocessing as preprocessing
import src.variables.global_vars as GV

class Analytics:
    def __init__(self) :
        pass
    
    # Runs MobileNet to get extrusion class
    def get_extrusion_class(self, img):
        extrusion_class = Mobilenet.run_onnx_inference(GV.ort_session, img)
        return extrusion_class
    
    def transform_img(self, img):
        img = Mobilenet.apply_transforms(img)
        return img
    
    def apply_gmms(self, img):
        img = preprocessing.gmms_preprocess_image(img, 6)
        return img