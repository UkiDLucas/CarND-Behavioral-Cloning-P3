
# coding: utf-8

# In[1]:

import argparse
import base64
import json
import numpy as np
import time
import eventlet
import eventlet.wsgi
import tensorflow as tf
import socketio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import numpy as np
import random
import math
from matplotlib import pyplot as plt

from numpy import array
import random
import scipy.misc


# In[2]:

# Define RGB colors used in the code
RED = color=[255, 0, 0]
GREEN = color=[0, 255, 0]
BLUE = color=[0, 0, 255]
WHITE = color=[255, 255, 255]
GRAY = color=[192, 192, 192]
VIOLET = color=[153, 51, 255]
ORANGE = color=[255, 128, 0] 


# # Read image from the disk

# In[3]:

def read_image_binary(image_path):
    """
    Returns:
    <class 'PIL.JpegImagePlugin.JpegImageFile'>
    """
    from PIL import Image
    image = Image.open(image_path)
    return image

def read_image_array(image_path):
    import cv2
    # cv2.IMREAD_COLOR 
    # cv2.COLOR_BGR2GRAY 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #print("image shape", image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return image


# In[4]:

import PIL
import numpy
from PIL import Image

def resize_image_maintain_ratio(numpy_array_image, new_size):
    """
    I am NOT zero-padding at this moment, 
    just resizing for the longest size is equal to new_size.
    The zero-padding can effectively by done later,
    for example during machine learning.
    There is no point of wasting space with
    thens of thousands padded padded images. 
    """
    # convert nympy array image to PIL.Image
    image = Image.fromarray(numpy.uint8(numpy_array_image))
    old_width = float(image.size[0])
    old_height = float(image.size[1])
    
    if old_width > old_height:
        # by width since it is longer
        new_width = new_size
        ratio = float(new_width / old_width)
        new_height = int(old_height * ratio)
    else:
        # by height since it is longer
        new_width = new_size
        ratio = float(new_width / old_width)
        new_height = int(old_height * ratio)
        
    image = image.resize((new_width, new_height), PIL.Image.ANTIALIAS)
    # turn image into nympy array again
    return array(image)


# In[5]:

def mask_vertices(image):
    """

    """
    height = image.shape[0]
    width = image.shape[1]

    top_left = (width*0.3, 0)
    top_right = (width-width*0.3, 0)
     
    mid_left_high = (0, height*0.2) 
    mid_right_high = (width, height*0.2)  
    
    mid_left_low = (0, height*0.9) 
    mid_right_low = (width, height*0.9)
    
    # on the bottom start high because of the dashboard
    bottom_center_left = (width*0.27, height*0.95) 
    bottom_center_right = (width-width*0.27, height*0.95) 
    
    # we are NOT following a center line in this code, so cut it out
    bottom_center = (width/2, height*0.55) 


    # add points clockwise
    vertices = np.array([[
        top_left, 
        top_right, 
        mid_right_high, mid_right_low,
        bottom_center_right,
        bottom_center, bottom_center_left,
        mid_left_low, 
        mid_left_high 
    ]], dtype=np.int32)
    return vertices


# In[6]:

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[7]:

def canny(image, low_threshold=50, high_threshold=250): 
    # homework low_threshold=20, high_threshold=130
    """Applies the Canny transform"""
    return cv2.Canny(image, low_threshold, high_threshold)


# In[8]:

def gaussian_blur(image, kernel_size=5): # 5 
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# In[9]:

import cv2
def grayscale(image):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[10]:

def crop_image(image):
    left = 0
    upper = 70
    right = 320
    lower = 140 # 160 original
    image = image.crop((left, upper, right, lower))
    return image


# In[11]:

def print_image(image, should_plot, comment="my image"):
   if should_plot:
       print(comment, array(image).shape)
       plt.imshow(image, cmap='gray')
       plt.show()


# In[12]:

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


# In[13]:

def round_int(x):
    if x == float("inf") or x == float("-inf"):
        # return float('nan') # or x or return whatever makes sense
        return 1000
    return int(round(x))

def test_round_int():
    print(round_int(174.919753086))
    print(round_int(0))
    print(round_int(float("inf")))
    print(round_int(float("-inf")))


# In[14]:

def calc_x(slope, y, y_intercept):
    
    if math.isnan(slope): # vertical line cannot have a slope
        return float('nan')
    if slope == float('Inf') or slope == -float('Inf'):
        return float('nan')
    if y_intercept == float('Inf') or y_intercept == -float('Inf'):
        return float('nan')
    
    result = 0 # temp
    try:
        if slope == 0: # flat line
            slope = 0.01 # avoid division by zero, result will be a large number, almost flat line
        x = (y - y_intercept)/slope
        result = round_int(x)
    except ValueError:
        print("ValueError: calc_x That was no valid number.  slope", slope, "y", y, "y_intercept", y_intercept)
    return   result


# In[15]:

def calc_y_intercept(slope, x, y):
    return y - (x * slope)


# In[16]:

def calc_slope(x1, y1, x2, y2):        
    rise = y2 - y1
    
    run = x2 - x1
    try:
        slope = rise/run
        return slope
    except ZeroDivisionError:
        print("ZeroDivisionError: calc_slope the slope cannot be calculated for a VERTICAL LINE.")
    

# TEST
#print(calc_slope(-1, 2, 1, 3))
#print(calc_slope(2, 2, 1, 3))
#print(calc_slope(1, 2, 1, 3))


# In[17]:

def side(image, line):
    """
    This function determines if line
    should be procesed as "left", "right"
    or rejected entirely as irrelevant.

    side: LEFT, slope -0.923076923077
    side: RIGHT, slope 0.65
    """
    

    width = image.shape[1] # right of the image frame
    height = image.shape[0] # bottom of the image frame

    for x1,y1,x2,y2 in arrangeLineCoordinates(line):
        slope = calc_slope(x1,y1,x2,y2)
        intercept = calc_y_intercept(slope, x1, y1)
        # I am interested where is x (left, or right)
        # if you extend the line to the bottom of the image
        y2 = height
        x2 = calc_x(slope, y2 , intercept) 

        if (x2 < width/2 - width*0.1) and (-0.95 < slope < -0.15) : # LEFT negative
            return "left"
        elif  (width/2 + width*0.1 < x2 ) and (0.95 > slope > 0.15): # RIGHT positive
            # print("side: RIGHT, slope", slope)
            return "right"
        else:
            # print("irrelevant, slope", slope, "x2", x2)
            return "irrelevant" # the line extends off screen, to be tested


# In[18]:

def arrangeLineCoordinates(line):
    """
    This method enforces that given line,
    has x1, y1 on TOP
    and x2, y2 on the BOTTOM of the image.
    
    It is user responsibility to test
    if line is a valid object.
    I have no way to know what to return otherwise.
    """
    try:
        for x1,y1,x2,y2 in line:
            if y1 > y2:
                # print("WARNING y1 > y2 swapping the order")
                temp_x2 = x1
                temp_y2 = y1
                temp_x1 = x2
                temp_y1 = y2

                x1 = temp_x1
                x2 = temp_x2
                y1 = temp_y1
                y2 = temp_y2   
                line = np.array([[x1, y1, x2, y2]], np.int32)
    except ValueError:
        #print("Provided line has unexpected values", line)
        line = np.array([[0, 0, 0, 0]], np.int32)
    except TypeError:
        # Use this as visual clue that line is not correct
        #print("Provided line has unexpected type", type(line))
        line = np.array([[0, 0, 0, 0]], np.int32)
                
    return line


# In[19]:

def draw_lines(image, lines, color=WHITE, thickness=1):
    """   
    Lines are drown over the image, i.e. mutates the image.
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None: # no point processing is no lines were found
        for line in lines:
            try:
                if line is not None: # TypeError: 'NoneType' object is not iterable
                    line = arrangeLineCoordinates(line)
                    for x1,y1,x2,y2 in line:
                        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
            except ValueError:
                #print("Oops!  draw_lines", line)
                cv2.line(image, (0, 0), (0, 0), color, thickness)
            except TypeError:
                #print("Oops!  draw_lines", line)
                cv2.line(image, (0, 0), (0, 0), color, thickness)


# In[20]:

def hough_lines(image, rho=2, theta=np.pi/180, threshold=20, min_line_len=10, max_line_gap=5):
    """
    - rho ρ is the distance from the origin
    - theta θ is the angle
    - min_line_len minimum length of a line that will be created
    - max_line_gap maximum distance between segments that will be connected to a single line
    - threshold increasing(~ 50-60) will rule out the spurious lines.
    defines the minimum number of intersections in a given grid cell that are required to choose a line.)
    """
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None: # no point processing if no lines were found
        return image
    
    width = image.shape[1] # right of the image frame
    height = image.shape[0] # bottom of the image frame

    left_longest_line = 0
    right_longest_line = 0

    relevant_hough_lines_left = [] 
    relevant_hough_lines_right = [] 
    rejected_hough_lines = []
    longest_lines_left = []
    longest_lines_right = []

    longest_right = 0
    longest_left = 0

    for line in lines:
        for x1,y1,x2,y2 in arrangeLineCoordinates(line):

            # get vertical HEIGHT of this line 
            y_difference = abs(y2 - y1)

            # Categorize the lines to LEFT | RIGHT 
            side_detected = side(image, line)
            
            if "left" == side_detected:
                relevant_hough_lines_left.append(line)
                if y_difference > longest_left:
                    left_longest_line = line
                    longest_left = y_difference
                    
            elif  "right" == side_detected:
                relevant_hough_lines_right.append(line)
                if y_difference > longest_right:
                    right_longest_line = line
                    longest_right = y_difference

            else:
                rejected_hough_lines.append(line) # WHITE

    longest_lines_left.append(left_longest_line)  # ORANGE 
    longest_lines_right.append(right_longest_line) # ORANGE 
    
    # draw a blank black image
    image_lines = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # draw color-coded HOUGH lines
    # Most of the time I do not want to draw all of the WHITE lines   
    #draw_lines(image_lines, lines, color=WHITE, thickness=2)
    draw_lines(image_lines, relevant_hough_lines_left, color=RED, thickness=1)
    draw_lines(image_lines, relevant_hough_lines_right, color=GREEN, thickness=1)

    return image_lines


# TEST  
#image_hough_lines = hough_lines(image_mask)
#plt.imshow(image_hough_lines)
#plt.show()


#image_hough_lines = hough_lines(image_mask, rho=2, theta=np.pi/180, threshold=25, min_line_len=25, max_line_gap=20)
#plt.imshow(image_hough_lines)
#plt.show()


# In[21]:

def preprocessing_pipline(image, final_size=512, should_plot=False):
    """
    final_size=256 AlexNet and GoogLeNet
    final_size=224 VGG-16
    final_size=64  is OPTIMAL if I was writing CDNN from scratch
    final_size=32  images are fuzzy, AlexNet (street signs CDNN)
    final_size=28  images are very fuzzy, LeNet
    """
    import matplotlib.pyplot as plt
    
        
    image = array(crop_image(image)) # 'numpy.ndarray' object has no attribute 'crop'
    print_image(image, should_plot, comment="my image")

    #image = region_of_interest(image, mask_vertices(image))
    #print_image(comment="grayscale", image, should_plot)
    
    image = grayscale(image)
    print_image(image, should_plot, comment="grayscale")

    image = gaussian_blur(image, kernel_size=5)
    print_image(image, should_plot, comment="gaussian_blur")
    
    image = canny(image, low_threshold=100, high_threshold=190)
    print_image(image, should_plot, comment="canny")
    
    image = hough_lines(image)
    print_image(image, should_plot, comment="hough_lines")
    
    image = resize_image_maintain_ratio(image, new_size=final_size)
    print_image(image, should_plot, comment="resize_image_maintain_ratio")

        
    image = normalize_grayscale(image)
    print_image(image, should_plot, comment="normalize_grayscale")

    return image

