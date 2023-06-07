"""
This module provides functionality for basic drawing operations.
"""

import cv2
import math

def draw(image_path, x, y, a, b):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    return image

def custom_draw(image, x, y, a, b):
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    return image

def draw_rectangle(image_path, x, y, a, b):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    cv2.imshow('', image)
    cv2.waitKey(10000)
    
# Theta is passed in degrees
def tip_line(img, theta, tip):
    angle_vector = [math.cos((theta*math.pi)/180)*50, math.sin((theta*math.pi)/180)*50] 
    angle_vector = [int(i) for i in angle_vector]
    img = cv2.line(img, [int(tip[0]), int(tip[1])], [int(tip[0])+angle_vector[0], int(tip[1])-angle_vector[1]], (255, 0, 0), 3)
    return img