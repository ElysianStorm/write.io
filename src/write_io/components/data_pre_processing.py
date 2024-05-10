import os
import urllib.request as request
import zipfile
from write_io.entity.config_entity import DataPreProcessingConfig
from write_io import logger
from write_io.utils.common import get_size
from pathlib import Path

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.config = config
        csv_path_training = self.file_path_training
        csv_path_validation = self.file_path_validation
        self.train = pd.read_csv(csv_path_training)
        self.valid = pd.read_csv(file_path_validation)

    def csv_cleanup(self):
        self.train.dropna(axis=0, inplace=True)
        self.valid.dropna(axis=0, inplace=True)
        unreadable = self.train[self.train['IDENTITY'] == 'UNREADABLE']
        unreadable.reset_index(inplace = True, drop=True)

        self.train = self.train[train['IDENTITY'] != 'UNREADABLE']
        self.valid = self.valid[valid['IDENTITY'] != 'UNREADABLE']

        self.train['IDENTITY'] = self.train['IDENTITY'].str.upper()
        self.valid['IDENTITY'] = self.valid['IDENTITY'].str.upper()

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
        train_size = 30000
        valid_size= 3000

        train_x = []

        for i in range(train_size):
            img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/'+train.loc[i, 'FILENAME']
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image/255.
            train_x.append(image)

            valid_x = []

        for i in range(valid_size):
            img_dir = '/kaggle/input/handwriting-recognition/validation_v2/validation/'+valid.loc[i, 'FILENAME']
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image/255.
            valid_x.append(image)

        train_x = np.array(train_x).reshape(-1, 256, 64, 1)
        valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
          
    
