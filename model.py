
# coding: utf-8

# # Train Model
# 
# 

# ## Set parameters

# In[1]:

data_dir = "../_DATA/CarND_behavioral_cloning/r_001/"
driving_data_csv = "driving_log_normalized.csv"
model_dir = "../_DATA/MODELS/"
model_name = "model_p3_14x64x3_"
batch_size = 256
nb_epoch = 40 
model_to_continue_training = "previous_model.h5"
previous_trained_epochs = 30

import os
from os.path import normpath, join

import scipy
from scipy import ndimage
from scipy.misc import imresize

import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import model_from_json, Sequential
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU as elu
from keras.layers import Flatten, ZeroPadding2D, MaxPooling2D, Activation, Dropout, Convolution2D
from keras.layers import Dense, Input, Activation, BatchNormalization, Lambda, ELU
from keras.optimizers import Adam
from keras.backend import ndim
# from keras.utils.visualize_util import plot

import csv
import cv2
import math
import json
import pickle
import random
import collections
import numpy as np

#import pydot

import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
tf.python.control_flow_ops = tf
# 

# In[3]:

#### Allocate only a fraction of memory to TensorFlow GPU process
# https://github.com/aymericdamien/TensorFlow-Examples/issues/38#issuecomment-265599695
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) # try range from 0.333 ot .9
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

#### Show available CPU and GPU(s)
from tensorflow.python.client import device_lib
def get_available_CPU_GPU():
    devices = device_lib.list_local_devices()
    #return [x.name for x in devices if x.device_type == 'CPU']
    return [x.name for x in devices ]

print(get_available_CPU_GPU())


# In[ ]:



