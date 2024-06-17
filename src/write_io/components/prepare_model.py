from write_io.entity.config_entity import PrepareModelConfig
from write_io import logger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

class PrepareModel:
    def __init__(self, config_model: PrepareModelConfig):
        input_data = Input(shape=(256, 64, 1), name='input')

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)
        
        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
        inner = Dropout(0.3)(inner)
        
        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
        inner = Dropout(0.3)(inner)
        
        # CNN to RNN
        inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
        
        ## RNN
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)
        
        ## OUTPUT
        inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)
        
        model = Model(inputs=input_data, outputs=y_pred)
        model.summary()

        labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        
        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

    
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
       
