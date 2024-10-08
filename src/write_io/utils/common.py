import os
from box.exceptions import BoxValueError
import yaml
from yaml import Loader
from write_io import logger
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox Type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.load(yaml_file, Loader=yaml.Loader)
            logger.info(f"yaml file: {path_to_yaml} load successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def append_to_yaml(file_path, data_to_append):
    """appends to yaml file

    Args: 
        file_path : yaml path input
        data_to_append : data to add to yaml file
    """
    # Load existing data from the YAML file
    try:
        with open(file_path, mode='r', encoding='utf-8') as yaml_file:
            existing_data = yaml.load(yaml_file, Loader=yaml.Loader)
    except FileNotFoundError:
        existing_data = {}

    # Append new data to the existing data
    if isinstance(existing_data, dict) and isinstance(data_to_append, dict):
        existing_data.update(data_to_append)
    elif isinstance(existing_data, list) and isinstance(data_to_append, list):
        existing_data.extend(data_to_append)
    else:
        raise ValueError("Existing data and data to append must be of the same type (dict or list).")

    # Write updated data back to the YAML file
    with open(file_path, mode='wt', encoding='utf-8') as yaml_file:
        yaml.dump(existing_data, yaml_file, default_flow_style=False)
        
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories

    Args:
        path_to_directories (list) : list of path to directories
        ignore_log (bool, optional) : ignore if multiple dirs is to be created. Defaults to False.
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    """ Save JSON data

    Args:
        path (Path): Path to JSON file
        data (dict): Data to be saved in JSON file
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """ Load JSON files data

    Args:
        path (Path): Path to JSON file
    
    Return:
        ConfigBox: Data as class attributes instead of dict
    """
    
    with open(path) as f:
        content = json.load(f)

    logger.info(f"JSON file loaded succesfully from: {path}")
    return ConfigBox(content)

def save_bin(data: Any, path: Path):
    """save Binary Tree

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

def load_bin(path:Path) -> Any:
    """Load Binary Data

    Args: 
        path (Path): path to binary file

    Returns:
        Any: Object stored in the file
    """

    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file
    
    Returns:
        str: size in KB
    """

    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())