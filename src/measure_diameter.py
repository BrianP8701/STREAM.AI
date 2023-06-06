import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def draw_line(img, slope, intercept):
    def equation(x): return int(slope * x + intercept)
    pt1 = equation(0)
    pt2 = equation(len(img[0])-1)
    return cv2.line(img, [0, pt1], [len(img[0])-1, pt2], (255, 0, 0), 1)

def rad_to_deg(degree):
    return (degree * math.pi) / 180

def whiten(img: np.ndarray):
    x = np.std(img)
    rang = 2.04023 * x - 4.78237
    if(x < 20): rang = 5
    whitestPixel = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel): whitestPixel = img[i][j]
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel - rang): img[i][j] = 255
    cutoff = 255 - rang
    while(True):    
        whitestPixel = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] < cutoff and img[i][j] > whitestPixel): whitestPixel = img[i][j]
        if whitestPixel == 0: break
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > whitestPixel - rang and img[i][j] < cutoff): img[i][j] = whitestPixel
        cutoff = whitestPixel - rang
    return img

def get_pixels_of_line(img, slope, intercept):
    allPixels = []
    Y = len(img)
    X = len(img[0])
    xstep = 1
    if abs(slope) > 1:
        xstep = 1 / abs(slope)
    ystep = slope * xstep
    x = 0
    y = intercept
    while(True):
        if(y < 0 or x >= X): break
        if(y < Y): allPixels.append([int(y), int(x)])
        x = x + xstep
        y = y + ystep
    return allPixels

def diagonal(pixel, img): return img[pixel[0]-1][pixel[1]+1] - img[pixel[0]+1][pixel[1]-1] == 0
def horizontal(pixel, img): return img[pixel[0]-1][pixel[1]] - img[pixel[0]+1][pixel[1]] == 0
def vertical(pixel, img): return img[pixel[0]][pixel[1]-1] - img[pixel[0]][pixel[1]+1] == 0

def get_confidence(img, all_pixels, slope):
    if abs(slope) < 0.7: 
        def func(pixel, img): return vertical(pixel, img)
    elif abs(slope) < 1.428: 
        def func(pixel, img): return diagonal(pixel, img)
    else:
        def func(pixel, img): return horizontal(pixel, img)
    confidence = 0
    for pixel in all_pixels:
        if(pixel[0] == 0 or pixel[0] == len(img)-1 or pixel[1] == 0 or pixel[1] == len(img[0])-1): continue
        if(not func(pixel, img)): confidence = confidence + 1
    return confidence

def get_center(img):
    return np.array([int(len(img[0]) / 2), int(len(img) / 2)])

def distance_from_point_to_line(slope, intercept, point):
    x = point[0]
    y = point[1]
    distance = abs((slope * x) - y + intercept) / math.sqrt(1 + (slope ** 2))
    return distance

def line_to_line(slope, intercept1, intercept2):
    return abs(intercept2 - intercept1) / math.sqrt((slope * slope) + 1)

def draw_rectangle(image, tip):
    cv2.rectangle(image, (tip[0][0]-2, tip[0][1]-2), (tip[0][0]+3, tip[0][1]+3), (255, 0, 0), 1)
    cv2.rectangle(image, (tip[1][0]-2, tip[0][1]-2), (tip[1][0]+3, tip[0][1]+3), (255, 0, 0), 1)
    cv2.imshow('', image)
    cv2.waitKey(10000)

def tip_outline(mTp, img):
    width = round(2.890 * mTp - 23.763)
    height = round(width * 2.5)
    
    kingOfTheHill = [0,0]
    highestConfidence = 0
    for x in range(0, len(img[0])-width-10):
        for y in range(0, len(img)-height-8):
            matrix = img[y:y+height+6, x:x+width+8]
            confidence = 0
            confidenceMultiple = 2.5
            # Vertical
            for i in range(0, 2):
                confidenceMultiple += 1
                for j in range(0, height):
                    if(matrix[j][0 + i] != matrix[j][8 - i]): confidence += confidenceMultiple
                    if(matrix[j][width-4+i] != matrix[j][width+4-i]): confidence += confidenceMultiple
                    
            confidenceMultiples = [6, 4, 4, 4, 2]
            # Horizontal
            for i in range(0, 5): 
                confidenceMultiple = confidenceMultiples[i]
                for j in range(4, width-4):
                    if(matrix[height-1-i][j] != matrix[height-12+i][j]): confidence += confidenceMultiple
            if(confidence > highestConfidence): 
                kingOfTheHill = [[x+5, y+height],[x+width+3, y+height]]
                highestConfidence = confidence
    return kingOfTheHill

def measure_diameter(img, theta, tip):
    if(theta >= 90):
        img = np.flip(img, 1)
        theta = 180 - theta
    slope = -math.tan(rad_to_deg(theta)) 
    slope_adjustment = 0.05 * slope # Trying slightly different slopes than given
    slope = slope - slope_adjustment
    highestConfidence = [0,0]
    pair = [[],[]] # slope, y intercept
    jump = -slope            # How much we change intercept by for each iteration
    if(jump < 1): jump = 1
    center = get_center(img)
    for q in range(0, 3):
        tip_intercept = tip[1] - slope * tip[0]
        intercept = 10
        highestconf = [0,0]
        curr_pair = [[],[]]
        last_line_intercept = -1000
        while(True):
            # IF line too close, only add if high conf. Otherwise continue to next
            this_lines_pixels = get_pixels_of_line(img, slope, intercept)
            if(len(this_lines_pixels) == 0): break 
            confidence = get_confidence(img, this_lines_pixels, slope)
            tip_multiple = distance_from_point_to_line(slope, intercept, tip)
            tip_multiple = -0.009*tip_multiple**2+0.084*tip_multiple+9.846
            if(tip_multiple < 1): tip_multiple = 1
            confidence *= tip_multiple
            if(line_to_line(slope, intercept, last_line_intercept) > 4):
                if(confidence > highestconf[0]):
                    highestconf[1] = highestconf[0]
                    curr_pair[1] = curr_pair[0]
                    highestconf[0] = confidence
                    curr_pair[0] = [slope, intercept]
                    last_line_intercept = intercept
                elif(confidence > highestconf[1]):
                    highestconf[1] = confidence
                    curr_pair[1] = [slope, intercept]
                    last_line_intercept = intercept
            intercept = intercept + jump
        slope = slope + slope_adjustment
        if(sum(highestconf) / 2 > sum(highestConfidence) / 2): 
            highestConfidence = highestconf
            pair = curr_pair
    return pair

def whiten(img: np.ndarray):
    x = np.std(img)
    rang = 2.04023 * x - 4.78237
    if(x < 20): rang = 5
    whitestPixel = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel): whitestPixel = img[i][j]
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel - rang): img[i][j] = 255
    cutoff = 255 - rang
    while(True):    
        whitestPixel = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] < cutoff and img[i][j] > whitestPixel): whitestPixel = img[i][j]
        if whitestPixel == 0: break
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > whitestPixel - rang and img[i][j] < cutoff): img[i][j] = whitestPixel
        cutoff = whitestPixel - rang
    return img


image_path = '/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/edgeimg/Edge7.jpeg'
image = cv2.imread(image_path)[:,:,0]
whiten(image)

pair = measure_diameter(image, 45, [53,45])
#image = np.flip(image, 1)
slope = pair[0][0]
intercept = pair[0][1]
allpixels = get_pixels_of_line(image, slope, intercept)
for pixel in allpixels:
    image[pixel[0]][pixel[1]] = 0

slope = pair[1][0]
intercept = pair[1][1]
allpixels = get_pixels_of_line(image, slope, intercept)
for pixel in allpixels:
    image[pixel[0]][pixel[1]] = 0

image = np.dstack([image, image, image])
plt.imshow(image, interpolation='nearest')
plt.show()
#image = draw_line(image, slope, intercept)



# python -c "import torch; model = torch.load('/Users/brianprzezdziecki/Downloads/yolov8s.pt'); torch.onnx.export(model, torch.randn(1, 3, 224, 224), 'model.onnx', input_names=['input'], output_names=['output'])"
