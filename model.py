def get_custom_model():

    import keras.backend as K
    from keras.models import Sequential
    from keras.activations import relu, softmax
    from keras.optimizers import SGD
    from keras.layers import ELU, InputLayer, Input
    from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.convolutional import ZeroPadding2D, Convolution1D, Cropping2D

    model = Sequential()

    # Crop the image
    input_shape = (160, 320, 3)
    top_crop = 70 # rows on the top of the image
    bottom_crop = 25 # hood of the car
    left_crop = 0
    right_crop = 0
    
    cropping=((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping, input_shape=input_shape))
    
    # normalize RGB 0-255 to -0.5 to 0.5
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(24, (5, 5), padding="same", activation="relu", data_format="channels_last"))
    model.add(Conv2D(36, (5, 5), padding="same", activation="relu"))
    #model.add(Conv2D(48, (3, 3), padding="same", activation="relu"))
    #model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Flatten())
    #model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2x2"))

    #model.add(Dense(100, activation="relu")) #256
    #model.add(Dropout(0.25))
    model.add(Dense(50, activation="relu" )) #256
    model.add(Dense(10, activation="relu" )) #256

    # CLASSIFICATION
    #model.add(Dense(41, activation='linear' , name="dense_3_41_linear")) 
    # default: linear | softmax | relu | sigmoid

    # REGRESSION
    model.add(Dense(1, activation='linear'))
    return model