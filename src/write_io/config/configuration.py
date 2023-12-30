from write_io.constants import *
from write_io.utils.common import read_yaml, create_directories
from write_io.entity.config_entity import DataIngestionConfig

# The ConfigurationManager is responsible for managing all the configuration details such as:
# Data Ingestion Configuration and more
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)
        # self.params = read_yaml(params_filepath)

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