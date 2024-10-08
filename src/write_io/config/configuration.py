from email.mime import base
from logging import root
from venv import create
import os

from write_io.constants import *
from write_io.utils.common import read_yaml, create_directories
from write_io.entity.config_entity import (DataIngestionConfig,
                                           DataPreProcessingConfig,
                                           PrepareBaseModelConfig,
                                           BuildModelConfig,
                                           TrainingModelConfig,
                                           PrepareCallbacksConfig,
                                           ValidationModelConfig)

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
            train_size = self.params.TRAIN_SIZE,
            validation_size = self.params.VALID_SIZE,
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

    def build_model_config(self) -> BuildModelConfig:
        model_config = self.config.build_model

        create_directories([model_config.root_dir])

        build_model_config = BuildModelConfig(
            root_dir = Path(model_config.root_dir),
            model_path = Path(model_config.model_path),
            updated_model_path = Path(model_config.updated_model_path),
            params_batch_size = self.params.BATCH_SIZE,
            params_epochs = self.params.EPOCHS,
            params_learning_rate = self.params.LEARNING_RATE,
            params_image_size = self.params.IMAGE_SIZE
        )

        return build_model_config
    
    def prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        callback_config = PrepareCallbacksConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath = Path(config.checkpoint_model_filepath)
        )

        return callback_config

    def training_model_config(self) -> TrainingModelConfig:
        training_path_config = self.config.training
        built_model_config = self.config.build_model.updated_model_path
        params = self.params
        training_data = self.config.data_pre_processing.image_path_training
        
        create_directories([
            Path(training_path_config.root_dir)
        ])

        training_model_config = TrainingModelConfig(
            root_dir=Path(training_path_config.root_dir),
            trained_model_path=Path(training_path_config.trained_model_path),
            updated_model_path=Path(built_model_config),
            training_data=Path(training_data),
            params_batch_size=params.BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_model_config
    
    def validation_model_config(self) -> ValidationModelConfig:
       training_config = self.config.build_model
       preprocessing_config = self.config.data_pre_processing
       base_model_config = self.config.prepare_base_model
       validation_model_config = ValidationModelConfig(
           trained_model_path=Path(training_config.model_path),
           file_path_validation = preprocessing_config.file_path_validation,
           alphabets = base_model_config.alphabets,
           valid_size=self.params.VALID_SIZE
       )

       return validation_model_config