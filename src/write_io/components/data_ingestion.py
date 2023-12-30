import os
import urllib.request as request
import zipfile
from write_io.entity.config_entity import DataIngestionConfig
from write_io import logger
from write_io.utils.common import get_size
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not  os.path.exists(self.config.local_data_file):
            # filename, headers = request.urlretrieve(
            #     url = self.config.source_URL,
            #     filename = self.config.local_data_file
            # )

            # logger.info(f"{filename} dowloaded with following information: \{headers}")
            pass
        else:    
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
            zip_file_path: str
            Extracts the zip file into the data directory defined in unzip_path
            Function returns None
        """

        unzip_path = self.config.local_data_file
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.source_URL, 'r') as zip_ref:
            zip_ref.extractall((unzip_path))