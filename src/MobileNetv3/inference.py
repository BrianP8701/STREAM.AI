'''
    This file contains the code for loading the model and performing inference on a single image.
    
    First load the model using the load_model function. 
    Then, passing that model and the path to an image to the infer_image function will return the predicted class.
    
    The output is a number starting from 0. These typically correspond to your classes alphabetically.
'''
from PIL import Image
from torchvision import transforms
import numpy as np
# You may need to adjust the input size based on the model you are using
# For example, EfficientNet-B0 uses 224, but other versions may use larger sizes
input_size = 224
preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
# Perform inference using ONNX runtime
def run_onnx_inference(session, input_image):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image.numpy()})[0]
    probabilities = softmax(outputs)
    predicted_class = np.argmax(probabilities) # 0, 1 or 2
    if predicted_class.item() == 0:
        return 'normal'
    elif predicted_class.item() == 1:
        return 'over'
    else:
        return 'under'
    
def apply_transforms(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        img = Image.fromarray((img * 255).astype(np.uint8))
        
    image = preprocess(img)
    image = image.unsqueeze(0)  # create a mini-batch as expected by the model
    
    return image

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()