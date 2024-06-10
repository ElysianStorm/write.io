import os
from write_io.entity.config_entity import DataPreProcessingConfig
from write_io import logger
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.config = config
        csv_path_training = config.file_path_training
        csv_path_validation = config.file_path_validation
        self.train = pd.read_csv(csv_path_training)
        self.valid = pd.read_csv(csv_path_validation)

        self.img_train = config.image_path_training
        self.img_valid = config.image_path_validation
        

    def csv_cleanup(self):
        self.train.dropna(axis=0, inplace=True)
        self.valid.dropna(axis=0, inplace=True)
        unreadable = self.train[self.train['IDENTITY'] == 'UNREADABLE']
        unreadable.reset_index(inplace = True, drop=True)

        self.train = self.train[self.train['IDENTITY'] != 'UNREADABLE']
        self.valid = self.valid[self.valid['IDENTITY'] != 'UNREADABLE']

        self.train['IDENTITY'] = self.train['IDENTITY'].str.upper()
        self.valid['IDENTITY'] = self.valid['IDENTITY'].str.upper()
        logger.info(f"CSV data cleaned.")
        logger.info("1. Rows with 'UNREADABLE' removed.")
        logger.info("All strings converted to UPPERCASE")

    def reformat_csv(self):    
        self.train.reset_index(inplace = True, drop=True) 
        self.valid.reset_index(inplace = True, drop=True)

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
    
    def image_normalization(self):
        train_size = self.config.train_size
        valid_size= self.config.validation_size

        train_x = []

        for i in range(train_size):
            img_dir = self.img_train+self.train.loc[i, 'FILENAME']
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            image = self.image_cleanup(image)
            image = image/255.
            train_x.append(image)

            valid_x = []

        for i in range(valid_size):
            img_dir = self.img_valid+self.valid.loc[i, 'FILENAME']
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            image = self.image_cleanup(image)
            image = image/255.
            valid_x.append(image)

        train_x = np.array(train_x).reshape(-1, self.config.resize_width, self.config.resize_height, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
        logger.info(f"IMAGE NORMALIZED.")
          
    
