'''
    This file contains the code for loading the model and performing inference on a single image.
    
    First load the model using the load_model function. 
    Then, passing that model and the path to an image to the infer_image function will return the predicted class.
    
    The output is a number starting from 0. These typically correspond to your classes alphabetically.
'''
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# You may need to adjust the input size based on the model you are using
# For example, EfficientNet-B0 uses 224, but other versions may use larger sizes
input_size = 224

preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, num_classes=3):
    # Initialize the model architecture
    model = models.mobilenet_v3_large
    model = model(num_classes)
    
    # Reset final fully connected layer
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def infer_image_simple(img):
    model = load_model('src2/MobileNetv3/mob_l_gmms2_finetune.pt')
    
    if isinstance(img, str):
        image = Image.open(img)
        
    image = preprocess(image)
    image = image.unsqueeze(0)  # create a mini-batch as expected by the model
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)

    return predicted_class.item()

def infer_image(img, model):    
    if isinstance(img, str):
        image = Image.open(img)
        
    image = preprocess(image)
    image = image.unsqueeze(0)  # create a mini-batch as expected by the model
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)

    return predicted_class.item()