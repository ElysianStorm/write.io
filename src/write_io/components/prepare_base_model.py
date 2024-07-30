from write_io.entity.config_entity import PrepareBaseModelConfig
from write_io.entity.config_entity import DataPreProcessingConfig
from write_io import logger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PrepareBaseModel:
    def __init__(self, config_base_model: PrepareBaseModelConfig, config_preprocessed_data: DataPreProcessingConfig):
        self.config_base_model = config_base_model
        self.alphabets = config_base_model.alphabets
        self.max_str_len = config_base_model.max_str_len # max length of input labels
        self.num_of_characters = config_base_model.num_of_characters # +1 for ctc pseudo blank
        self.num_of_timestamps = config_base_model.num_of_timestamps # max length of predicted labels

        # Get Processed Data
        self.config_preprocessed_data = config_preprocessed_data
        csv_path_training = config_preprocessed_data.file_path_training
        csv_path_validation = config_preprocessed_data.file_path_validation
        self.train = pd.read_csv(csv_path_training)
        self.valid = pd.read_csv(csv_path_validation)
        self.img_train = config_preprocessed_data.image_path_training
        self.img_valid = config_preprocessed_data.image_path_validation

    def label_to_num(self,label):
        # Convert characters to numbers
        label_num = []
        for ch in label:
            label_num.append(self.alphabets.find(ch))
            
        return np.array(label_num)

    def num_to_label(self,num):
        # Convert numbers to characters
        ret = ""
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=self.alphabets[ch]
        return ret
            
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

        training_data_yaml = readTrainingDataYaml()
        training_data_yaml['train_y'] = train_y
        training_data_yaml['train_input_len'] = train_input_len
        training_data_yaml['train_label_len'] = train_label_len
        training_data_yaml['train_output'] = train_output
        training_data_yaml['valid_y'] = valid_y
        training_data_yaml['valid_input_len'] = valid_input_len
        training_data_yaml['valid_label_len'] = valid_label_len
        training_data_yaml['valid_output'] = valid_output
        
        updateTrainingYaml(training_data_yaml)

        logger.info(f"Training data processed for base model.")
