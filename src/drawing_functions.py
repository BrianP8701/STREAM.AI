"""
This module provides functionality for basic drawing operations.
"""
import cv2
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw(img, x, y, a, b):
    if isinstance(img, str):
        img = cv2.imread(img)
    cv2.rectangle(img, (x, y), (a, b), (255, 0, 0), 1)
    cv2.imshow('', img)
    cv2.waitKey(0)
    return img

def show(img_path):
    # Load the image
    img = mpimg.imread('/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2104/frame11.jpg')

    # Display the image
    plt.imshow(img)
    plt.show()
    
def draw_return(image, x, y, a, b):
    cv2.rectangle(image, (x, y), (a, b), (255, 0, 0), 1)
    return image

# Theta is passed in degrees
def tip_line(img, theta, tip):
    angle_vector = [math.cos((theta*math.pi)/180)*50, math.sin((theta*math.pi)/180)*50] 
    angle_vector = [int(i) for i in angle_vector]
    img = cv2.line(img, [int(tip[0]), int(tip[1])], [int(tip[0])+angle_vector[0], int(tip[1])-angle_vector[1]], (255, 0, 0), 3)
    return img