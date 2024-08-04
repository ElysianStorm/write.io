from dataclasses import dataclass
from pathlib import Path

# Defining the object of data that will be ingested/used for the project  
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreProcessingConfig:
    train_size: int
    validation_size: int
    resize_width: int
    resize_height: int
    file_path_training: Path
    file_path_validation: Path
    image_path_training: Path
    image_path_validation: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    alphabets: str
    max_str_len: int
    num_of_characters: int
    num_of_timestamps: int

@dataclass(frozen=True)
class BuildModelConfig:
    root_dir: Path
    model_path: Path
    params_learning_rate: float
    params_batch_size: int
    params_epochs: int
    params_image_size: list

@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    trained_model_path: Path
    updated_model_path: Path
    training_data: Path

    params_learning_rate: float
    params_batch_size: int
    params_epochs: int
    params_image_size: list 

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path