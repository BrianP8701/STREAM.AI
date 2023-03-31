import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import numpy.linalg as linalg

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

def point_to_line(slope, intercept, point):
    p2 = np.array([0, intercept])
    p1 = np.array([100, intercept + slope * 100])
    return np.cross(p2-p1,point-p1)/linalg.norm(p2-p1)

def line_to_line(slope, intercept1, intercept2):
    return abs(intercept2 - intercept1) / math.sqrt((slope * slope) + 1)

def measure_diameter(img, theta):
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
        intercept = 10
        highestconf = [0,0]
        curr_pair = [[],[]]
        last_line_intercept = -1000
        while(True):
            # IF line too close, only add if high conf. Otherwise continue to next
            this_lines_pixels = get_pixels_of_line(img, slope, intercept)
            if(len(this_lines_pixels) == 0): break
            confidence = get_confidence(img, this_lines_pixels, slope)
            confidenceMultiple = abs(1 / point_to_line(slope, intercept, center))
            if(confidenceMultiple > 0.3): confidenceMultiple = 0.05
            confidence = confidence * (1 + confidenceMultiple*3)
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

image_path = 'Measure_Extraction/edgeimg/Edge10.jpeg'
image = cv2.imread(image_path)[:,:,0]
whiten(image)

image = np.dstack([image, image, image])
plt.imshow(image, interpolation='nearest')
plt.show()

pair = measure_diameter(image, 18)

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

#image = draw_line(image, slope, intercept)



# Take the center. Being closer to center increases confidence.
# Lines too close get ignored