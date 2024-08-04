from write_io import logger
from write_io.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from write_io.pipeline.stage_02_data_preprocessing import DataPreProcessingPipeline
from write_io.pipeline.stage_03_prep_base_model import BaseModelPrepPipeline
from write_io.pipeline.stage_04_build_model import BuildModelPipeline
from write_io.pipeline.stage_05_training_model import TrainingModelPipeline

try:
    # STAGE_NAME = "Data Ingestion Stage"
    # logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    # obj = DataIngestionTrainingPipeline()
    # obj.main()
    # logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")

    STAGE_NAME = "Data Pre Processing Stage"
    logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    obj = DataPreProcessingPipeline()
    obj.main()
    logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")

    STAGE_NAME = "Preparing Base Model Stage"
    logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    obj = BaseModelPrepPipeline()
    obj.main()
    logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")

    STAGE_NAME = "Building Model Stage"
    logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    obj = BuildModelPipeline()
    obj.main()
    logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")

    
    STAGE_NAME = "Training Stage"
    logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    obj = TrainingModelPipeline()
    obj.main()
    logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")

except Exception as e:
    logger.exception(e)
    raise e
    
