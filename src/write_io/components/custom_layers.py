import tensorflow as tf
from keras import backend as K

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, labels, input_length, label_length = inputs
        y_pred = y_pred[:, 2:, :]  # Skip initial garbage outputs
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    def get_config(self):
        config = super(CTCLayer, self).get_config()
        return config
