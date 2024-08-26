from write_io.entity.config_entity import TestModelConfig
from write_io.constants import *
from write_io.utils.common import read_yaml, save_json
import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
from write_io.components.custom_layers import CTCLayer


class TestModelConfig:
    def __init__(self, test_model_config: TestModelConfig):
        self.config = test_model_config
        self.alphabets = self.config.alphabets
        self.image_path = self.config.image_path

    def get_model(self):
        # Load the model with the custom layer
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path,
            custom_objects={'CTCLayer': CTCLayer}
        )

    def test_model(self):
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        image = self.image_cleanup(image)
        image = image/255.

        pred = self.model.predict(image.reshape(1, 256, 64, 1))
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                        greedy=True)[0][0])
        
        print(f"Predicted: " + self.num_to_label(decoded[0]))


    def num_to_label(self,num):
        # Convert numbers to characters
        ret = ""
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=self.alphabets[ch]
        return ret
    
    def image_cleanup(self,img):
        (h, w) = img.shape
        resize_h = self.config.resize_height 
        resize_w = self.config.resize_width
        final_img = np.ones([resize_h, resize_w])*255 # blank white image
        
        # crop
        if w > resize_w:
            img = img[:, :resize_w]
            
        if h > resize_h:
            img = img[:resize_h, :]
        
        final_img[:h, :w] = img
        return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)