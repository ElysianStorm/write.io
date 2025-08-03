# ✍️ write.io: A Deep Dive into a Handwriting Recognition Model

---

## 📑 Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Implementation](#implementation)
- [Project Pipeline & Inter-Component Flow](#project-pipeline--inter-component-flow)
- [Detailed File Explanations](#detailed-file-explanations)
- [Conclusion](#conclusion)

---

## 📘 Overview

The **write.io** repository provides a comprehensive and modular pipeline for building a handwriting recognition system. The project's workflow is managed by a centralized `ConfigurationManager` which reads settings from [`config.yaml`](config.yaml) and hyperparameters from [`params.yaml`](params.yaml).

The core of the system is a **Convolutional Recurrent Neural Network (CRNN)** designed to interpret handwritten text from images. This architecture is particularly suited for text recognition tasks due to its ability to extract spatial features from images (via CNNs) and then process these features as a sequence (via RNNs). A crucial aspect of the model is its use of a custom [`CTCLayer`](custom_layers.py) and the **Connectionist Temporal Classification (CTC)** loss function, which efficiently handles the challenge of aligning variable-length text predictions with fixed-size input images. 

The pipeline is an end-to-end process, starting with raw data and concluding with a trained model and a set of performance metrics, including **character** and **word accuracy**, derived from a validation dataset.

---

## 🚦 Approach

The project's methodology is a **systematic, stage-based workflow**, designed for clarity and reproducibility. The entire process is orchestrated by [`main.py`](main.py), which executes each stage of the pipeline sequentially:

1. **Data Ingestion**: [`data_ingestion.py`](data_ingestion.py), configured by [`configuration.py`](configuration.py), downloads and extracts the data from the `source_URL` in [`config.yaml`](config.yaml).

2. **Data Preprocessing**: Involves cleanup and normalization to ensure the image and label data are formatted correctly for training.

3. **Label Preparation**: Handled by [`prepare_base_model.py`](prepare_base_model.py), this converts ground truth text labels into numerical representations with uniform length (for CTC loss).

4. **Model Building**: Constructed by [`build_model.py`](build_model.py), using CNN + BiLSTM layers with custom [`CTCLayer`](custom_layers.py).

5. **Model Training**: [`training_model.py`](training_model.py) handles training with callbacks defined in [`prepare_callbacks.py`](prepare_callbacks.py), using hyperparameters from [`params.yaml`](params.yaml).

6. **Validation**: [`validation_model.py`](validation_model.py) loads the trained model and evaluates it on validation data using character and word accuracy metrics.

<img width="1024" height="1536" alt="ChatGPT Image Aug 3, 2025, 09_18_09 PM" src="https://github.com/user-attachments/assets/6cc5e58a-f275-4519-8ad5-0741f09a4bcc" />


---

## 🛠 Implementation

The project’s implementation reflects a **modular design**, with each component performing a specific, well-defined task. The entire process is a seamless flow of data and control, orchestrated from a single entry point: [`main.py`](main.py).

---

## 🔁 Project Pipeline & Inter-Component Flow

Execution is driven by [`main.py`](main.py):

### 📌 Orchestration
- Initializes `ConfigurationManager` from [`configuration.py`](configuration.py).

### 📌 Configuration & Parameter Loading
- Uses `read_yaml()` from [`common.py`](common.py) to load settings from:
  - [`config.yaml`](config.yaml)
  - [`params.yaml`](params.yaml)
- Sets up directory structure using `create_directories()`.
- Instantiates config entities from [`config_entity.py`](config_entity.py).

### 📌 Stage Execution

Each stage is executed in sequence:

1. [`stage_01_data_ingestion.py`](stage_01_data_ingestion.py): Downloads and extracts data using `DataIngestion`.

2. [`stage_02_data_preprocessing.py`](stage_02_data_preprocessing.py): Cleans and normalizes data.

3. [`stage_03_prep_base_model.py`](stage_03_prep_base_model.py): Uses `PrepareBaseModel` to process text labels for CTC.

4. [`stage_04_build_model.py`](stage_04_build_model.py): Builds CRNN and adds `CTCLayer`.

5. [`stage_05_training_model.py`](stage_05_training_model.py): Trains model using `TrainingModelConfig` and training callbacks.

6. [`stage_06_validation.py`](stage_06_validation.py): Evaluates the trained model and saves final metrics.

---

## 📂 Detailed File Explanations

- [`config.yaml`](config.yaml): Main config file; defines file paths, URLs, and project structure.

- [`params.yaml`](params.yaml): Stores model hyperparameters like `EPOCHS`, `BATCH_SIZE`, and `IMAGE_SIZE`.

- [`config_entity.py`](config_entity.py): Contains data classes (`DataIngestionConfig`, etc.) for type-safe configuration.

- [`configuration.py`](configuration.py): Central `ConfigurationManager` for loading config and parameter YAMLs.

- [`common.py`](common.py): Utility functions like `read_yaml`, `create_directories`, and `save_json`.

- [`custom_layers.py`](custom_layers.py): Defines custom `CTCLayer` that implements CTC loss.

- [`main.py`](main.py): Entry point; runs all pipeline stages in order.

- [`build_model.py`](build_model.py): Builds the CRNN model architecture and appends `CTCLayer`.

- [`training_model.py`](training_model.py): Loads data and model, trains it using callbacks and hyperparameters.

- [`validation_model.py`](validation_model.py): Loads the trained model, makes predictions, computes final metrics.

---

## 🧾 Conclusion

The **write.io** project is a robust, well-engineered solution for handwriting recognition, demonstrating a **best-practice** approach to building a machine learning pipeline. The **CRNN architecture**, combined with the **CTC loss function** via a custom layer, is a powerful and effective choice for this domain.

With a modular pipeline, driven by a centralized `ConfigurationManager` and external YAML files, the project is transparent, highly configurable, and easy to maintain. It provides a **complete, end-to-end solution**, from raw data ingestion to final validation and performance evaluation of the model.
