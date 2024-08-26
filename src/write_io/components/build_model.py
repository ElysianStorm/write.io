from keras.layers import *
from keras.models import Model
from pathlib import Path
import tensorflow as tf
import tensorflow as tf
from keras import backend as K
from keras.saving import register_keras_serializable
from keras.optimizers import Adam
from write_io.components.custom_layers import CTCLayer

@register_keras_serializable()
def ctc_loss_function(y_true, y_pred):
    return y_pred  # y_pred already contains the CTC loss computed by the CTCLayer

class BuildModel:
    def __init__(self, model_config, config):
        self.model_config = model_config
        self.config = config
        self.input_data = Input(shape=self.model_config.params_image_size, name='input')
        self.labels = Input(name='gtruth_labels', shape=[self.config.max_str_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

    def get_model(self):
        self.model = self.prepareBasicModel()
        self.save_model(path=self.model_config.model_path, model=self.model)
        self.final_model = self.prepare_model_last_stage()
        self.save_model(path=self.model_config.updated_model_path, model=self.final_model)

    def prepareBasicModel(self):
         # Conv2D: 32 filters, kernel size 3x3, 'same' padding.
        # BatchNormalization: Normalizes the output.
        # Activation: ReLU.
        # MaxPooling2D: Pool size 2x2.
        # Reduces the spatial dimensions and extracts features.
        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(self.input_data)  
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
        self.y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=self.input_data, outputs=self.y_pred)
        model.summary()

        return model

    def prepare_model_last_stage(self):
        # Defining the loss function as a separate layer since its a custom loss function/layer
        ctc_loss = CTCLayer(name='ctc')([self.y_pred, self.labels, self.input_length, self.label_length])
        model_final = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=ctc_loss)
        model_final.compile(loss=ctc_loss_function, optimizer=Adam(learning_rate=self.model_config.params_learning_rate))
        model_final.summary()

        return model_final

    @staticmethod
    def save_model(path, model):
        model.save(path)