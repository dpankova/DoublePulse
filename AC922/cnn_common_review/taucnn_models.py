# Functions for building different tau ID models
#
# Aaron Fienberg
# April 2020

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Activation, concatenate
from tensorflow.keras.regularizers import l2

# default l2 regularization value
l2_val = 0.0025


def RGB_original():
    '''first RGB model
    based heavily on the original, single string model that started it all
    '''
    model = keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(50, 10),
                     activation='relu', input_shape=(500, 60, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(25, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(15, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    return model

#
# Multi input models
#
def make_multi_input_vars_adapter():
    ''' generates an input adapter for models with multi-channel inputs
    '''
    def adapter(img_array, var_array):
        #print(img_array.shape,var_array.shape)
        return [img_array[:,:,:,:1],img_array[:,:,:,1:2],\
                img_array[:,:,:,2:3],var_array[:,0]]

    return adapter


vars_adapter = make_multi_input_vars_adapter()



def make_multi_input_adapter(n_channels):
    ''' generates an input adapter for models with multi-channel inputs
    '''
    def adapter(img_array):
        return [img_array[:, :, :, i:i+1] for i in range(n_channels)]

    return adapter


three_chan_adapter = make_multi_input_adapter(3)


def CNN_fan_in(n_channels=3):
    '''runs each channel through the same CNN in parallel,
       concatenates the outputs, and then runs those through
       a few dense layers.

       Uses batch normalization in the dense layers
    '''

    CNN_layers = Sequential(name='convolutional_layers')
    CNN_layers.add(Conv2D(filters=32, kernel_size=(50, 10),
                          activation='relu', input_shape=(500, 60, 1)))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(25, 5), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_layers.add(Conv2D(filters=16, kernel_size=(15, 3), activation='relu'))
    CNN_layers.add(AveragePooling2D(pool_size=(4, 4)))
    CNN_layers.add(Flatten())

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [CNN_layers(channel) for channel in channels]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=64, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=1, activation='sigmoid'))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)


def CNN_fan_in_regularized(n_channels=3):
    ''' Like fan_in, but with l2 regularization and 
        dropout used in the final dense layers
    '''
    CNN_layers = Sequential(name='convolutional_layers')
    CNN_layers.add(Conv2D(filters=32, kernel_size=(50, 10),
                          activation='relu', input_shape=(500, 60, 1)))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(25, 5), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2)))
    CNN_layers.add(Conv2D(filters=16, kernel_size=(15, 3), activation='relu'))
    CNN_layers.add(AveragePooling2D(pool_size=(4, 4)))
    CNN_layers.add(Flatten())

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [CNN_layers(channel) for channel in channels]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False,
                           kernel_regularizer=l2(l2_val)))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=64, use_bias=False,
                           kernel_regularizer=l2(l2_val)))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=1, activation='sigmoid',
                           kernel_regularizer=l2(l2_val)))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)


def parallel_CNNs(n_channels=3):
    ''' like fan-in, but with independent weights for each CNN

       Uses batch normalization in the dense layers
    '''

    CNN_layers = []
    for i in range(n_channels):
        CNN = Sequential(name=f'convolutional_layers_{i+1}')
        CNN.add(Conv2D(filters=32, kernel_size=(50, 10),
                       activation='relu', input_shape=(500, 60, 1)))
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Conv2D(filters=64, kernel_size=(25, 5), activation='relu'))
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Conv2D(filters=16, kernel_size=(15, 3), activation='relu'))
        CNN.add(AveragePooling2D(pool_size=(4, 4)))
        CNN.add(Flatten())
        CNN_layers.append(CNN)

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [network(channel)
                  for network, channel in zip(CNN_layers, channels)]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=64, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=1, activation='sigmoid'))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)

def CNN_vgg16_vars(n_channels=3):
    '''Runs each channel through the same CNN in parallel,
       concatenates the outputs, and then runs those through
       a few dense layers.

       Uses batch normalization in the dense layers
    '''

    CNN_layers = Sequential(name='convolutional_layers')
    CNN_layers.add(ZeroPadding2D((1,1),input_shape=(500, 60, 1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    CNN_layers.add(Flatten())

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [CNN_layers(channel) for channel in channels]
    exvars = Input(shape=(7))
    cnn_outputs = concatenate(cnn_stages)
    all_outputs = concatenate([cnn_outputs,exvars])

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    dense_layers.add(Dense(units=64, use_bias=False))
    dense_layers.add(Dense(units=1, activation='sigmoid'))

    output = dense_layers(all_outputs)
    return Model(inputs=[channels]+[exvars], outputs=output)


def CNN_vgg16(n_channels=3):
    '''Runs each channel through the same CNN in parallel,
       concatenates the outputs, and then runs those through
       a few dense layers.

       Uses batch normalization in the dense layers
    '''

    CNN_layers = Sequential(name='convolutional_layers')
    CNN_layers.add(ZeroPadding2D((1,1),input_shape=(500, 60, 1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    CNN_layers.add(Flatten())

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [CNN_layers(channel) for channel in channels]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    #dense_layers.add(BatchNormalization(scale=False))
    #dense_layers.add(Activation('relu'))
    #dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=64, use_bias=False))
    #dense_layers.add(BatchNormalization(scale=False))
    #dense_layers.add(Activation('relu'))
    #dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=1, activation='sigmoid'))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)

def CNN_vgg16_muon(n_channels=3):
    '''Runs each channel through the same CNN in parallel,
       concatenates the outputs, and then runs those through
       a few dense layers.

       Uses batch normalization in the dense layers
    '''

    CNN_layers = Sequential(name='convolutional_layers')
    CNN_layers.add(ZeroPadding2D((1,1),input_shape=(500, 60, 1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(ZeroPadding2D((1,1)))
    CNN_layers.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    CNN_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    CNN_layers.add(Flatten())

    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    cnn_stages = [CNN_layers(channel) for channel in channels]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=64, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dropout(rate=0.5))
    dense_layers.add(Dense(units=3, activation='softmax'))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)


def Xception_fan_in(n_channels=3):
    '''runs each channel through the same Xception model in parallel,
       concatenates the outputs, and then runs those through
       a few dense layers.
       
       Uses batch normalization in the dense layers
    '''
    
    xception_layers = Sequential(name='xception_layers') 
    xception_layers.add(keras.layers.ZeroPadding2D((0, 6),
                                                    input_shape=(500,60,1)))
    xception_layers.add(
        keras.applications.Xception(include_top=False,
        weights=None,
        input_shape=(500,72,1),
        pooling=None)
    )
    xception_layers.add(Flatten())
    
    channels = [Input(shape=(500, 60, 1)) for _ in range(n_channels)]
    
    cnn_stages = [xception_layers(channel) for channel in channels]
    cnn_outputs = concatenate(cnn_stages)

    dense_layers = Sequential(name='dense_layers')
    dense_layers.add(Dense(units=128, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=64, use_bias=False))
    dense_layers.add(BatchNormalization(scale=False))
    dense_layers.add(Activation('relu'))
    dense_layers.add(Dense(units=1, activation='sigmoid'))

    output = dense_layers(cnn_outputs)

    return Model(channels, output)
