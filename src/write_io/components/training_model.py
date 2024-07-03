from write_io.entity.config_entity import PrepareBaseModelConfig
from write_io.entity.config_entity import DataPreProcessingConfig, BuildModelConfig
from write_io import logger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as keras
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout


class PrepareBaseModel:
    def __init__(self,  model_config: BuildModelConfig, config_base_model: PrepareBaseModelConfig, config_preprocessed_data: DataPreProcessingConfig):
        pass
        # model_final = keras.models.load_model(self.model_config.model_path)

        # # the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
        # model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = self.model_config.params_learning_rate))

        # model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
        #                 validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
        #                 epochs=60, batch_size=128)