
# coding: utf-8

# # Train Model
# 
# 

# ## Set parameters that will control the execution

# In[15]:

data_dir = "../_DATA/CarND/p3_behavioral_cloning/set_000/"
image_dir = "IMG/"
driving_data_csv = "driving_log_original.csv"
YIELD_BATCH_SIZE = 32 #256
RUN_EPOCHS = 3 

should_retrain_existing_model = False
saved_model = "model_epoch_33_val_acc_0.0.h5"
previous_trained_epochs = 0


# In[2]:

import DataHelper


# # Allocate only a fraction of memory to TensorFlow GPU process

# In[3]:

# https://github.com/aymericdamien/TensorFlow-Examples/issues/38#issuecomment-265599695
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) # try range from 0.3 to 0.9
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

#### Show available CPU and GPU(s)
from tensorflow.python.client import device_lib
def get_available_CPU_GPU():
    devices = device_lib.list_local_devices()
    #return [x.name for x in devices if x.device_type == 'CPU']
    return [x.name for x in devices ]

print(get_available_CPU_GPU())


# # Fetch data from CSV file

# In[4]:

from  DataHelper import read_csv
csv_path = data_dir + driving_data_csv
print("csv_path", csv_path)
headers, data = read_csv(data_dir + driving_data_csv)


# # Split data into training, testing and validation sets

# In[5]:

from DataHelper import split_random
training, testing, validation = split_random(data, percent_train=75, percent_test=15) 

print("training", training.shape)
print("testing", testing.shape)
print("validation", validation.shape)


# # Fetch and visualize training steering angles
# 
# I would like to train a car on the set that has a nice bell curve distribution of values:
# - I can drive the car on the track backwards
# - I can flip each image (and value)

# In[ ]:




# # Remove zero-steering angles from training set 

# In[6]:

import numpy as np
from DataHelper import plot_histogram, get_steering_values, find_nearest


# In[7]:

def remove_zeros(training):
    
    print("len(training)", len(training))
    indexes_to_keep = []
    
    steering_angles = get_steering_values(training)
    plot_histogram("steering values", steering_angles, change_step=0.01)

    for index in range (len(steering_angles)):
        angle = steering_angles[index]
        if angle != 0: 
            indexes_to_keep.append(index)

    print("len(indexes_to_keep)", len(indexes_to_keep))

    training_to_keep = []
    for index in indexes_to_keep:
        training_to_keep.append(training[index])

    training = training_to_keep
    # release the memory
    training_to_keep = []
    indexes_to_keep = []

    print("len(training)", len(training))

    steering_angles = get_steering_values(training)
    plot_histogram("steering values", steering_angles, change_step=0.01)
    return training

training = remove_zeros(training)


# # Extract image names

# In[8]:

def get_center_image_names(training):
    from DataHelper import get_image_center_values 
    image_names = get_image_center_values(training)
    print("image count", image_names.shape[0])
    print(image_names[1])
    return image_names

image_names = get_center_image_names(training)


# # Create a list of image paths

# In[9]:

def build_image_paths(image_names):
    image_paths = []
    for image_name in image_names:
        image_paths.extend([data_dir + image_name])
    print(image_paths[1]) 
    print("found paths:", len(image_paths) ) 
    return image_paths

image_paths = build_image_paths(image_names)


# # Read actual images

# In[10]:

def read_images(image_paths):
    import numpy as np 
    from ImageHelper import read_image_array
    #training_features = [read_image_array(path) for path in image_paths]

    image_list = []
    for path in image_paths[0:5]:
        image_list.append(read_image_array(path))
    training_features = np.array(image_list) # numpy array, not just a list
    return training_features


# In[11]:

training_features = read_images(image_paths)

print ("image_paths[2]", image_paths[2] )
print ("training_features count", len(training_features) )

sample_image = training_features[2]
print ("sample_image  ", sample_image.shape)

import matplotlib.pyplot as plt
plt.imshow(sample_image) # cmap='gray' , cmap='rainbow'
plt.show()

#print(sample_image[0][0:15])


# # Define yield Generator

# In[21]:

def generator(training, batch_size=32):
    """
    Yields batches of training and testing data every time the generator is called.
    """
    import sklearn
    import numpy as np
    while 1: # Loop forever so the generator never terminates
        shuffle(training)
        for offset in range(0, len(training), batch_size):
            training_batch = training[offset:(offset + batch_size)]
            
            steering_angles = get_steering_values(training_batch)
            
            image_names = get_center_image_names(training_batch)
            image_paths = build_image_paths(image_names)
            training_features = read_images(image_paths)


            # trim image to only see section with road
            X_train = np.array(training_features)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(training, YIELD_BATCH_SIZE)
validation_generator = generator(testing, YIELD_BATCH_SIZE)


# In[12]:

image_dimentions = (3, 80, 320)  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=image_dimentions,
        output_shape=image_dimentions))
#model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=RUN_EPOCHS)


# # Import Keras (layer above TensorFlow)
# 
# https://keras.io/layers/convolutional/

# In[ ]:

import keras.backend as K
from keras.models import Sequential
from keras.layers import ELU, InputLayer, Input
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
from keras.activations import relu, softmax
from keras.optimizers import SGD
import cv2, numpy as np
from DataHelper import mean_pred, false_rates
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D


# # Build a Convolutional Neural Network

# ## Minimal Model

# In[ ]:

def get_CDNN_model_minimal(input_shape):
    model = Sequential()
    
    model.add(Lambda(lambda x: x/255.0 - 0.5, # normalize RGB 0-255 to -0.5 to 0.5
                     input_shape=input_shape,
                    name="Normalize_RGB"))
    model.add(Convolution2D(32, 3, 3, border_mode='same', 
                            activation="relu", dim_ordering='tf', name="Convo_ReLU_32x3x3_01"))
    model.add(Convolution2D(32, 5, 5, border_mode='same', 
                            activation="relu", name="Convo_ReLU_32x5x5_02" ))
    model.add(Convolution2D(32, 5, 5, border_mode='same', 
                            activation="relu", name="Convo_ReLU_32x5x5_03" ))
    model.add(Flatten())
    #model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2x2"))

    model.add(Dense(256, activation="relu", name="Dense_relu_256_01")) #256
    model.add(Dropout(0.25, name="Dropout_0.25_01"))
    model.add(Dense(256, activation="relu", name="Dense_relu_256_02" )) #256

    # CLASSIFICATION
    #model.add(Dense(41, activation='linear' , name="dense_3_41_linear")) # default: linear | softmax | relu | sigmoid

    # REGRESSION
    model.add(Dense(1, activation='linear'))
    return model


# # Compile model (configure learning process)

# In[ ]:

input_shape = (160, 320, 3) # sample_image   (160, 320, 3)
model = get_CDNN_model_minimal(input_shape)
model.summary()
# Before training a model, you need to configure the learning process, which is done via the compile method.
# 
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

optimizer='sgd' # | 'rmsprop'
loss_function="mse" # | 'binary_crossentropy' | 'mse' | mean_squared_error | sparse_categorical_crossentropy
metrics_array=['accuracy'] # , mean_pred, false_rates

model.compile(optimizer, loss_function, metrics_array)


# # Replace model with one stored on disk
# 
# - If you replace the model, the INPUT dimetions have to be the same as these trained
# - Name your models well
from keras.models import load_model

if should_retrain_existing_model:
    model_path = model_dir + model_to_continue_training
    model = load_model(model_path) 
    model.summary()
# # Train (fit) the model agaist given labels

# In[ ]:

print( "training_features.shape", len(training_features) )
# REGRESSION
history = model.fit(training_features, 
                    y = steering_angles, 
                    nb_epoch = nb_epoch, 
                    batch_size = batch_size, 
                    verbose = 2, 
                    validation_split = 0.2)

# CLASSIFICATION
#history = model.fit(training_features, 
#y_one_hot, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.2)

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
conv2d_1_relu (Convolution2D)    (None, 160, 320, 32)  896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1638400)       0           conv2d_1_relu[0][0]              
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             1638401     flatten_1[0][0]                  
====================================================================================================
Total params: 1,639,297
Trainable params: 1,639,297
Non-trainable params: 0

training_features.shape 590
Train on 472 samples, validate on 118 samples
- MacBook Pro CPU 13s / epoch
- MacBook Pro GPU 5s / epoch



____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
conv2d_1_relu (Convolution2D)    (None, 160, 320, 32)  896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
conv2d_3_relu (Convolution2D)    (None, 160, 320, 32)  25632       conv2d_1_relu[0][0]              
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1638400)       0           conv2d_3_relu[0][0]              
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             1638401     flatten_1[0][0]                  
====================================================================================================
Total params: 1,664,929
Trainable params: 1,664,929
Non-trainable params: 0


training_features.shape 590
Train on 472 samples, validate on 118 samples
- MacBook Pro CPU 114s / epoch
- MacBook Pro GPU 29s / epoch (29%)

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
conv2d_1_relu (Convolution2D)    (None, 160, 320, 32)  896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
conv2d_2_relu (Convolution2D)    (None, 160, 320, 32)  9248        conv2d_1_relu[0][0]              
____________________________________________________________________________________________________
conv2d_3_relu (Convolution2D)    (None, 160, 320, 32)  25632       conv2d_2_relu[0][0]              
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1638400)       0           conv2d_3_relu[0][0]              
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             1638401     flatten_1[0][0]                  
====================================================================================================
Total params: 1,674,177
Trainable params: 1,674,177
Non-trainable params: 0

Train on 472 samples, validate on 118 samples
- MacBook Pro CPU 156s / epoch
- MacBook Pro GPU ResourceExhaustedError
# In[ ]:

# list all data in history
print(history.history.keys())

training_accuracy = str( history.history['acc'][nb_epoch-1])
print("training_accuracy", training_accuracy)

training_error = str( history.history['loss'][nb_epoch-1])
print("training_error", training_error)

validation_accuracy = str( history.history['val_acc'][nb_epoch-1])
print("validation_accuracy", validation_accuracy)

validation_error = str( history.history['val_loss'][nb_epoch-1])
print("validation_error", validation_error)


# # Save the model

# In[ ]:

# creates a HDF5 file '___.h5'
model.save(data_dir 
           + "model_epoch_" + str(nb_epoch + previous_trained_epochs) 
           + "_val_acc_" + str(validation_accuracy) 
           + ".h5") 
#del model  # deletes the existing model
#model = load_model('my_model.h5')


# # summarize history for accuracy

# In[ ]:

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (bigger better)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'testing accuracy'], loc='lower right')
plt.show()



# # summarize history for loss

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Validation error (smaller better)')
plt.ylabel('error')
plt.xlabel('epochs run')
plt.legend(['training error(loss)', 'validation error (loss)'], loc='upper right')
plt.show()


# # Prediction

# In[ ]:

from keras.models import load_model

model_path = data_dir + saved_model
print(model_path)

model = load_model(model_path) 
model.summary()


# In[ ]:

image_name = "center_2016_12_01_13_38_59_461.jpg" # stering 0.05219137
original_steering_angle = 0.7315571

image_path =  data_dir +   image_name
print(image_path)
image = read_image(image_path)
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()


# ## Run model.predict(image)

# In[ ]:

predictions = model.predict( image[None, :, :], 
                            batch_size = 1, 
                            verbose = 1)


# # Extract top prediction

# In[ ]:

from DataHelper import predict_class

predicted_class = predict_class(predictions, steering_classes)

print("original steering angle \n", original_steering_angle)
print("top_prediction \n", predicted_class )


# # Plot predictions (peaks are top classes)

# In[ ]:

# summarize history for loss
plt.plot(predictions[0])
plt.title('predictions')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['predictions'], loc='upper right')
plt.show()


# In[ ]:



