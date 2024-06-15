from email.mime import base

from voluptuous import unicode
from write_io.constants import *
from write_io.utils.common import read_yaml, create_directories
from write_io.entity.config_entity import (DataIngestionConfig,
                                           DataPreProcessingConfig,
                                           PrepareBaseModelConfig,
                                           PrepareModelConfig)

# The ConfigurationManager is responsible for managing all the configuration details such as:
# Data Ingestion Configuration and more
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # Data Ingestion Configuration Manager is responsible for:
    # Coupling the data metadata/params defined in entity/config_entity.py within the project
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config

    def pre_processing_params_config(self) -> DataPreProcessingConfig:
        config = self.config.data_pre_processing

        data_pre_processing_config = DataPreProcessingConfig(
            train_size = config.train_size,
            validation_size = config.validation_size,
            resize_width = config.resize_width,
            resize_height = config.resize_height,
            file_path_training = config.file_path_training,
            file_path_validation = config.file_path_validation,
            image_path_training = config.image_path_training,
            image_path_validation = config.image_path_validation
        )

        return data_pre_processing_config
    
    def prepare_base_model_params_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        prepare_base_model_config = PrepareBaseModelConfig(
            alphabets = config.alphabets,
            max_str_len = config.max_str_len, # max length of input labels
            num_of_characters = len(config.alphabets) + 1, # +1 for ctc pseudo blank
            num_of_timestamps = config.num_of_timestamps # max length of predicted labels
        )

        return prepare_base_model_config

    def prepare_model_params_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        prepare_model_config = PrepareModelConfig(
            shape = config.shape, # (256, 64, 1) 
        )

        return prepare_model_config
    


    # def prepare_base_model(self) -> PrepareBaseModelConfig:
    #     config = self.config.prepare_base_model

    #     create_directories([config.root_dir])

    #     prepare_base_model_config = PrepareBaseModelConfig(
    #         root_dir = Path(config.root_dir),
    #         base_model_path= Path(config.base_model_path),
    #         updated_base_model_path= Path(config.updated_base_model_path),
    #         params_image_size= self.params.IMAGE_SIZE,
    #         params_learning_rate= self.params.LEARNING_RATE,
    #         params_include_top= self.params.INCLUDE_TOP,
    #         params_weights= self.params.WEIGHTS
    #     )

    #     return prepare_base_model_config
