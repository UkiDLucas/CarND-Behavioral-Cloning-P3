
# coding: utf-8

# In[1]:

"""
Helper methods are very specific to p3.
Created by Uki D. Lucas on Feb. 4, 2017
"""

should_run_tests = False


# In[2]:

def predict_class(predictions, classes):
    percentages = predictions[0]

    # reverse sort by value, return indexes
    sorted_indexes = (-percentages).argsort()
 
    for index in sorted_indexes[:1]:
        print("predicted class:", classes[index], 
              "\t",  np.round(percentages[index]*100,1), "%") 
    
    # right now we will return top prediction
    predicted_class = classes[sorted_indexes[0]]
    return predicted_class


# In[3]:

# http://localhost:8888/notebooks/ImageHelper.ipynb#Read-image-from-the-disk
def read_image(image_path):
    import cv2
    # cv2.IMREAD_COLOR 
    # cv2.COLOR_BGR2GRAY 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #print("image shape", image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return np.array(image)


# In[4]:

import numpy as np

# TODO implement batch_from, batch_to - I did not need it for 8037 rows
# TODO implement has_header_row
def read_csv(file_path):
    """
    Usage:
    headers, data = read_csv(file_path)
    Parameter: 
    - file_path: can be relative path "../../../DATA/stuff.csv"
    Returns:
    - headers: array of strings e.g. ['steering', 'throttle', 'brake', 'speed']
    - data: array of strings, you have to convert values to int, float yourself
   test_read_csv()
    """
    import csv
    # Opening spreadsheet to read in TEXT mode: 'rt'
    with open(file_path, 'rt') as csvfile:
        # Most common format of CSV, TODO improve
        payload = csv.reader(csvfile, delimiter=',', quotechar='"') 
        row_counter = 0
        headers = []
        data = []
        for row in payload:
            row_counter = row_counter + 1
            
            if row_counter == 1:
                headers = row
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
            elif 1 < row_counter < 3:
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
                data.append(row)
            else:
                data.append(row)
        print("Number of imported CSV rows:", row_counter)
        # I am returning data as numpy array instead of list
        # because it is handier to use it.
        return headers, np.array(data)
    
    
    
def test_read_csv():
    """
    This test is specific to Uki's enviroment.
    """
    data_dir = "../../../DATA/behavioral_cloning_data/"
    headers, data = read_csv(file_path = data_dir + 'driving_log.csv')
    print("headers \n",headers)
    print("3rd row of data \n",data[2:3])

if should_run_tests:    
    test_read_csv()


# In[5]:

def create_steering_classes(number_of_classes = 41):
    steering_classes = np.linspace(-1, 1, num=number_of_classes, endpoint=True, dtype=np.float32) 
    steering_classes = np.sort(steering_classes)
    return steering_classes


# In[6]:

# snapping actual values to given labels

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

if should_run_tests:  
    assert (find_nearest(steering_labels, -0.951) == -1.00),"method find_nearest() has problem"


# In[ ]:




# In[7]:

import numpy as np
import math
    
def split_random(matrix, percent_train=70, percent_test=15):
    """
    Splits matrix data into randomly ordered sets 
    grouped by provided percentages.
    
    Usage:
    rows = 100
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = \
    split_random(matrix, percent_train=80, percent_test=10)
    
    percent_validation 10
    training (80, 2)
    testing (10, 2)
    validation (10, 2)
    
    Returns:
    - training_data: percentage_train e.g. 70%
    - testing_data: percent_test e.g. 15%
    - validation_data: reminder from 100% e.g. 15%
    Created by Uki D. Lucas on Feb. 4, 2017
    """
    #print(matrix)  
    row_count = matrix.shape[0]
    np.random.shuffle(matrix)
    
    end_training = int(math.ceil(row_count*percent_train/100))    
    end_testing = end_training + int(math.ceil((row_count * percent_test/100)))
    
    percent_validation = 100 - percent_train - percent_test
    
    training = matrix[:end_training]
    testing = []
    validation = []
    
    if percent_validation < 0:
        print("Make sure that the provided sum of " +         "training and testing percentages is equal, " +         "or less than 100%.")
        
        testing = matrix[end_training:]
    else:
        print("percent_validation", percent_validation)
        
        testing = matrix[end_training:end_testing]
        validation = matrix[end_testing:]
    
    return training, testing, validation

# TEST:

def test_split_random():
    rows = 8037
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = split_random(matrix, percent_train=80, percent_test=20) 

    print("training",training.shape)
    print("testing",testing.shape)
    print("validation",validation.shape)
    
    print("sum",training.shape[0] + testing.shape[0])
    
    #print(split_random.__doc__)
if should_run_tests:  
    test_split_random()


# In[8]:

def get_image_center_values(matrix):
    data = [row[0] for row in matrix]
    return np.array(data)

def get_image_left_values(matrix):
    data = [row[1] for row in matrix]
    return np.array(data)

def get_image_right_values(matrix):
    data = [row[2] for row in matrix]
    return np.array(data)

def get_steering_values(matrix):
    data = [float(row[3]) for row in matrix]
    return np.array(data).astype('float32')

def get_throttle_values(matrix):
    data = [float(row[4]) for row in matrix]
    return np.array(data)

def get_brake_values(matrix):
    data = [float(row[5]) for row in matrix]
    return np.array(data)

def get_speed_values(matrix):
    data = [float(row[6]) for row in matrix]
    return np.array(data)


# In[ ]:




# In[9]:

# for custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def false_rates(y_true, y_pred):
    false_neg = ...
    false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }


# In[ ]:




# In[10]:

def sort_unique_floats(array_x):
    # assure that the array is numpy and numerical
    #array_x = array_x.astype(np.float)
    
    # get unique values, a.k.a. set of values
    labels_set = set(array_x)
    #print("labels_set\n", labels_set)
    
    # set is not sorted, so convert it to a numpy array
    unique_labels = np.array(list(labels_set))
    #print("unique_labels\n", unique_labels.shape, unique_labels)
    
    sorted_unique_labels = np.sort(unique_labels)
    #print("sorted_unique_labels\n", sorted_unique_labels.shape, sorted_unique_labels)
    return sorted_unique_labels
    
# TEST   


# In[11]:

def locate_one_hot_position(defined_classes, actual_label):
    
    #print("actual_label", actual_label, type(actual_label))
    matched_label = float( find_nearest(defined_classes, actual_label))
    
    #print("matched_label", matched_label, type(matched_label))
    found_at = np.where(defined_classes == matched_label)
    
    # (array([10]),) <class 'tuple'>
    found_at = found_at[0][0] # first element of first vector
    #print("found_at", found_at, type(found_at) )
    # returns array of locations
    return found_at

def encode_one_hot(defined_classes, sample_labels):
    """
    Creates one hot encoded vector from a list {1D vector (None,)} containing training labels.
    - find all unique labels
    - count all unique labels
    - create a zero filed array, size equal to count of all unique labels
    - order the unique values (small to large)
    - create empty output matrix
    - for each sample's label create zero vector and set one in position of that label
    Created by Uki D. Lucas
    """
    defined_classes = sort_unique_floats(defined_classes)
    
    # possible float rounding errors
    defined_classes = np.round_(defined_classes, decimals=1)
    print("defined_classes\n", defined_classes)
    
    class_count = len(defined_classes)
    print("class_count:", class_count)
    
    sample_count = len(sample_labels)
    print("sample_count:", sample_count)
    
    one_hot = np.zeros(shape=(sample_count, class_count), dtype=np.int)
     
    for index in range(sample_count): 
        actual_label = float(sample_labels[index])
        # find first index of actual_label
        found_at = locate_one_hot_position(defined_classes, actual_label)
        #print("found", index, type(index) )
        one_hot[index][found_at] = 1
        #print("one_hot[index]", one_hot[index])
    print("one_hot examples \n", one_hot[0:3])
    print("one_hot shape", one_hot.shape) 
    return one_hot


# In[12]:

def show_layers(model):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(i, ") ",layer.name, "\t\t is trainable: ", layer.trainable)
        #layer.trainable = False
    return len(model.layers)


# In[13]:

import numpy as np
#import random
import math
import matplotlib
from matplotlib import pyplot as plt

def margin(value):
    return value*(5/100)

def plot_histogram(name, values, change_step):
    
    min_value = min(values)
    print("min_value", min_value)
    max_value = max(values)
    print("max_value", max_value)
    
    spread = max_value-min_value
    print("spread", spread)
    recommended_bins = math.ceil(spread/change_step)
    print("recommended number of classes", recommended_bins)
    
    bins = np.linspace(math.floor(min(values)), 
                       math.ceil(max(values)),
                       recommended_bins)

    plt.xlim([
        min_value - margin(min_value), 
        max_value + margin(max_value)])

    plt.hist(values, bins=bins, alpha=0.5)
    plt.title('Distribution of ' + name)
    plt.xlabel('values')
    plt.ylabel('occurance')
    
    # RESIZE it nicely for Jupyter Notebook (width = 10)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 3)
    fig.savefig('test2png.png', dpi=72)
    plt.margins(0.1)
    
    plt.show()


# In[14]:

def plot_steering_values(values):
    
    import matplotlib.pyplot as plt
    
    plt.plot(values, 'b.')

    plt.title("Distribution of steering value classes.")
    plt.xlabel("class number")
    plt.ylabel('steering value')
    
    # RESIZE it nicely for Jupyter Notebook (width = 10)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 3)
    fig.savefig('test2png.png', dpi=72)
    plt.margins(0.1)
    
    plt.show()


# In[ ]:



