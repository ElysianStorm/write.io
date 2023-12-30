from write_io import logger
from write_io.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"------------- STAGE {STAGE_NAME} Completed -------------")
except Exception as e:
    logger.exception(e)
    raise e