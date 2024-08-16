from write_io.entity.config_entity import TrainingModelConfig
from write_io.constants import *
from write_io.utils.common import read_yaml
import tensorflow as tf
from write_io.components.custom_layers import CTCLayer

class TrainingModelConfig:
    def __init__(self, config: TrainingModelConfig):
        self.config = config
        training_data_file = TRAINING_DATA_FILE_PATH
        self.training_data = read_yaml(training_data_file)
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_model(self):
        # Load the model with the custom layer
        self.model = tf.keras.models.load_model(
            self.config.updated_model_path,
            custom_objects={'CTCLayer': CTCLayer}
        )

    def train_model(self):
        print("Training MODEL...")

        train_x = self.training_data['train_x']
        train_y = self.training_data['train_y']
        train_input_len = self.training_data['train_input_len']
        train_label_len = self.training_data['train_label_len']
        train_output = self.training_data['train_output']
        valid_x = self.training_data['valid_x']
        valid_y = self.training_data['valid_y']
        valid_input_len = self.training_data['valid_input_len']
        valid_label_len = self.training_data['valid_label_len']
        valid_output = self.training_data['valid_output']

        self.model.fit(
            x=[train_x, train_y, train_input_len, train_label_len],
            y=train_output,
            validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
            epochs=self.params.EPOCHS,
            batch_size=self.params.BATCH_SIZE
        )