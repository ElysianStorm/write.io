from write_io.entity.config_entity import TrainingModelConfig
from write_io import logger
from pathlib import Path
from write_io.constants import *
from write_io.utils.common import read_yaml
import tensorflow as tf
import keras as keras
from keras.optimizers import Adam


class TrainingModelConfig:
    def __init__(self,  config: TrainingModelConfig):
        self.config = config
        training_data = TRAINING_DATA_FILE_PATH
        self.training_data = read_yaml(training_data)
        
        print("GOT DATA")
        
    def get_model(self):
        print("GETTING MODEL...")
        self.model = tf.keras.models.load_model(
            self.config.updated_model_path,
            safe_mode=False
        )
        # self.final_model = self.prepare_model_last_stage()
        # self.save_model(path=self.model_config.model_path, model=self.final_model)

        print("MODEL GOT FOR TRAINING")

    def train_model(self):
        print("Training MODEL...")
        # # the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss        
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = self.model_config.params_learning_rate))

        train_x = self.training_data.train_x
        train_y = self.training_data.train_y
        train_input_len = self.training_data.train_input_len
        train_label_len = self.training_data.train_label_len
        train_output = self.training_data.train_output
        valid_x = self.training_data.valid_x
        valid_y = self.training_data.valid_y
        valid_input_len = self.training_data.valid_input_len
        valid_label_len = self.training_data.valid_label_len
        valid_output = self.training_data.valid_output

        self.model.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                        validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                        epochs=2, batch_size=128)
        
        print("MODEL TRAINED")