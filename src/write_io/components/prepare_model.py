from write_io.entity.config_entity import PrepareModelConfig
from write_io import logger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PrepareBaseModel:
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

    
    def process_labels(self):
        # Process characters to convert them into numbers and add padding to set each label to same length
    
        # train_y, valid_y contains the true labels converted to numbers and padded with -1. The length of each label is equal to max_str_len.
        # train_label_len, valid_label_len contains the length of each true label (without padding)
        # train_input_len, valid_input_len contains the length of each predicted label. The length of all the predicted labels is constant i.e number of timestamps - 2.
        # train_output, valid_output is a dummy output for ctc loss.
        
        # Process Labels for Training Data
        train_size = self.config_preprocessed_data.train_size
        valid_size = self.config_preprocessed_data.validation_size
               
        train_y = np.ones([train_size, self.max_str_len]) * -1
        train_label_len = np.zeros([train_size, 1])
        train_input_len = np.ones([train_size, 1]) * (self.num_of_timestamps-2)
        train_output = np.zeros([train_size])

        for i in range(train_size):
            train_label_len[i] = len(self.train.loc[i, 'IDENTITY'])
            train_y[i, 0:len(self.train.loc[i, 'IDENTITY'])]= self.label_to_num(self.train.loc[i, 'IDENTITY'])

        logger.info(f"Training data processed for base model.")

        # Process Labels for Validation Data
        valid_y = np.ones([valid_size, self.max_str_len]) * -1
        valid_label_len = np.zeros([valid_size, 1])
        valid_input_len = np.ones([valid_size, 1]) * (self.num_of_timestamps-2)
        valid_output = np.zeros([valid_size])

        for i in range(valid_size):
            valid_label_len[i] = len(self.valid.loc[i, 'IDENTITY'])
            valid_y[i, 0:len(self.valid.loc[i, 'IDENTITY'])]= self.label_to_num(self.valid.loc[i, 'IDENTITY'])  

        logger.info(f"Training data processed for base model.")
