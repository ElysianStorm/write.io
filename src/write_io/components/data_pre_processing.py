import os
import urllib.request as request
import zipfile
from write_io.entity.config_entity import DataPreProcessingConfig
from write_io import logger
from write_io.utils.common import get_size
from pathlib import Path

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.config = config

    def csv_cleanup(self):
        

    def image_cleanup(self):
          
    def reformat_csv(self):

    def image_normalization(self):
