import os
from pathlib import Path
import logging
# Template file used for creating directory template for Deep Learning Pipeline Projects
# Can be re-used to create template directory with all the folders, files that can allow you to get started with the project directly

# Configuring logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "write.io"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    # research/trials.ipynb is for scratchpad work during the project development.
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory {filedir} created")
        logging.info(f"File {filename} created in Directory {filedir}")

    if (not os.path.exists(filename)) or (os.path.getsize(filename)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")