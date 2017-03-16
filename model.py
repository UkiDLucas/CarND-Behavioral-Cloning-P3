def get_custom_model(input_shape):

    import keras.backend as K
    from keras.models import Sequential
    from keras.activations import relu, softmax
    from keras.optimizers import SGD
    from keras.layers import ELU, InputLayer, Input
    from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D

    model = Sequential()

    # TODO trim the image
    
    # normalize RGB 0-255 to -0.5 to 0.5
    model.add(Lambda(lambda x: x/255.0 - 0.5, 
                     input_shape=input_shape))




    model.add(Convolution2D(32, 3, 3, border_mode='same', activation="relu", dim_ordering='tf'))
    #model.add(Convolution2D(32, 5, 5, border_mode='same', activation="relu" ))
    #model.add(Convolution2D(32, 5, 5, border_mode='same', activation="relu" ))
    model.add(Flatten())
    #model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2x2"))

    model.add(Dense(256, activation="relu")) #256
    model.add(Dropout(0.25, name="Dropout_0.25_01"))
    #model.add(Dense(256, activation="relu" )) #256

    # CLASSIFICATION
    #model.add(Dense(41, activation='linear' , name="dense_3_41_linear")) # default: linear | softmax | relu | sigmoid

    # REGRESSION
    model.add(Dense(1, activation='linear'))
    return model