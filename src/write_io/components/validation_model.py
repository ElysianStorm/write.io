from write_io.entity.config_entity import ValidationModelConfig
from write_io.constants import *
from write_io.utils.common import read_yaml, save_json
import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
from write_io.components.custom_layers import CTCLayer

class ValidationModelConfig:
    def __init__(self, validation_model_config: ValidationModelConfig):
        self.config = validation_model_config
        training_data_file = TRAINING_DATA_FILE_PATH
        self.trained_data = read_yaml(training_data_file)
        self.trained_model_path = self.config.trained_model_path
        self.valid_size = self.config.valid_size

        csv_path_validation = self.config.file_path_validation
        self.valid = pd.read_csv(csv_path_validation)
        self.alphabets = self.config.alphabets

    def get_model(self):
        # Load the model with the custom layer
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path,
            custom_objects={'CTCLayer': CTCLayer}
        )

    def validate_model(self, callback_list):
        valid_x = self.trained_data['valid_x']
        self.preds = self.model.predict(valid_x, callbacks=callback_list)
        print(self.preds)
        decoded = K.get_value(K.ctc_decode(self.preds, input_length=np.ones(self.preds.shape[0])*self.preds.shape[1], greedy=True)[0][0])

        prediction = []
        for i in range(self.valid_size):
            prediction.append(self.num_to_label(decoded[i]))

        y_true = self.valid.loc[0:self.valid_size, 'IDENTITY']
        correct_char = 0
        total_char = 0
        correct = 0

        for i in range(self.valid_size):
            pr = prediction[i]
            tr = y_true[i]
            total_char += len(tr)
            
            for j in range(min(len(tr), len(pr))):
                if tr[j] == pr[j]:
                    correct_char += 1
                    
            if pr == tr :
                correct += 1 

        self.character_accuracy = correct_char*100/total_char
        self.word_accuracy = correct*100/self.valid_size
        print('Correct characters predicted : %.2f%%' %(self.character_accuracy))
        print('Correct words predicted      : %.2f%%' %(self.word_accuracy))

        self.save_scores()

    def num_to_label(self,num):
        # Convert numbers to characters
        ret = ""
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=self.alphabets[ch]
        return ret
    
    def save_scores(self):
        scores = {
            "character_accuracy": self.character_accuracy,
            "word_accuracy": self.word_accuracy
        }

        save_json(path=SCORES_PATH, data=scores)