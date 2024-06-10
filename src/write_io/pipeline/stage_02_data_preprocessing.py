from write_io.config.configuration import ConfigurationManager
from write_io.components.data_pre_processing import DataPreProcessing
from write_io import logger

STAGE_NAME = "Data Pre Processing stage"

class DataPreProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_pre_processing_config = config.pre_processing_params_config()
        data_pre_processing = DataPreProcessing(config=data_pre_processing_config)
        data_pre_processing.csv_cleanup()
        data_pre_processing.reformat_csv()
        data_pre_processing.image_normalization()

        # processed_data = (
        #     data_pre_processing.train,
        #     data_pre_processing.valid,
        #     data_pre_processing.img_train,
        #     data_pre_processing.img_valid
        # )
        # return processed_data

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = DataPreProcessingPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e
