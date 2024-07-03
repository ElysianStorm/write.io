from write_io.entity.config_entity import BuildModelConfig, PrepareBaseModelConfig
from keras.layers import *
from keras.models import Model
from pathlib import Path
import tensorflow as tf

class BuildModel:
    def __init__(self, model_config: BuildModelConfig, config: PrepareBaseModelConfig):
        self.model_config = model_config
        self.config = config

    def get_model(self):
        self.model = self.prepareModel(shape=self.model_config.params_image_size)
        self.save_model(path=self.model_config.model_path, model=self.model)

    def prepareModel(self, shape):
        # Shape: (256, 64, 1)
        # This is where the input data enters the model. The input is expected to be a 256x64 grayscale image.
        input_data = Input(shape=shape, name='input')

        # Conv2D: 32 filters, kernel size 3x3, 'same' padding.
        # BatchNormalization: Normalizes the output.
        # Activation: ReLU.
        # MaxPooling2D: Pool size 2x2.
        # Reduces the spatial dimensions and extracts features.
        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

        # Conv2D: 64 filters, kernel size 3x3, 'same' padding.
        # BatchNormalization: Normalizes the output.
        # Activation: ReLU.
        # MaxPooling2D: Pool size 2x2.
        # Dropout: 0.3 to prevent overfitting.
        # Further extracts features and reduces dimensions
        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
        inner = Dropout(0.3)(inner)

        # Conv2D: 128 filters, kernel size 3x3, 'same' padding.
        # BatchNormalization: Normalizes the output.
        # Activation: ReLU.
        # MaxPooling2D: Pool size 1x2.
        # Dropout: 0.3 to prevent overfitting.
        # Extracts deeper features and reduces dimensions.
        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
        inner = Dropout(0.3)(inner)

        # CNN to RNN
        # Reshape: Converts the output to shape (64, 1024).
        # Prepares the data for RNN processing by flattening spatial dimensions.
        inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)

        # Dense: 64 units, ReLU activation.
        # Adds a fully connected layer to integrate the features.
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

        ## RNN
        # Bidirectional LSTM (2 Layers): 256 units, returns sequences.
        # Processes the sequence data in both forward and backward directions.
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
        inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

        ## OUTPUT
        # Dense: 29 units (one for each character).
        # Activation: Softmax.
        # Outputs probabilities over the 29 possible character classes.
        inner = Dense(self.config.num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=input_data, outputs=y_pred)
        model.summary()
        return model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
