"""
This module provides functionality for basic drawing operations.
"""
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
    
def draw_return(image, x, y, a, b, color=(255, 0, 0), thickness=1):
    cv2.rectangle(image, (x, y), (a, b), color, thickness)
    return image


def tip_line(img, theta, tip):
    # Theta is passed in degrees
    angle_vector = [math.cos((theta*math.pi)/180)*50, math.sin((theta*math.pi)/180)*50] 
    angle_vector = [int(i) for i in angle_vector]
    img = cv2.line(img, [int(tip[0]), int(tip[1])], [int(tip[0])+angle_vector[0], int(tip[1])-angle_vector[1]], (255, 0, 0), 3)
    return img

def write_text_on_image(image_np, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 0), thickness=2):
    """
    Write text on the given image.
    
    Args:
    - image_np (numpy array): The image to write text on.
    - text (str): The text to write.
    - position (tuple): The position (x, y) where the text should start.
    - font (cv2 font): Font type. Default is Hershey Simplex.
    - font_scale (float): Font scale factor that is multiplied by the font-specific base size.
    - color (tuple): Color of the text in the format (B, G, R). Default is blue.
    - thickness (int): Thickness of the text stroke.

    Returns:
    - numpy array: The image with text written on it.
    """
    cv2.putText(image_np, text, position, font, font_scale, color, thickness)
    return image_np

def plot_points(points_list, destination_path, plot_title='Scatter Plot of Points', x_label='X values', y_label='Y values'):
    """
    Plots a scatter plot of the points provided in the points_list.
    
    Parameters:
    - points_list: List of [x,y] coordinates.
    - destination_path: Path to save the generated plot.
    - plot_title: Optional title for the plot.
    - x_label: Optional label for the x-axis.
    - y_label: Optional label for the y-axis.
    """
    
    # Extract x and y values from the list of lists
    x_values = [point[0] for point in points_list]
    y_values = [point[1] for point in points_list]

    # Plot the points
    plt.scatter(x_values, y_values, color='blue', marker='o')

    if len(points_list) > 0:
        # Setting the axis limits
        plt.xlim(min(x_values), max(x_values))
        plt.ylim(min(y_values), max(y_values))

    # Setting the title and labels using the provided arguments
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Saving the plot to the given destination path
    plt.savefig(destination_path)
    plt.close()
    
def draw_line(img, line):
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 1)
    return img
    