from write_io import logger
from write_io.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from write_io.pipeline.stage2_data_preprocessing.py import DataPreProcessingPipeline

STAGES = ["Data Ingestion Stage" , "Data Pre Processing Stage"]
for stage in STAGES:
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"------------- STAGE {STAGE_NAME} Completed -------------")
    except Exception as e:
        logger.exception(e)
        raise e
