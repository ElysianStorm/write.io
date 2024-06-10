from write_io.config.configuration import ConfigurationManager
from write_io.components.prepare_base_model import PrepareBaseModel
from write_io import logger

STAGE_NAME = "Prepare Base Model stage"

class BaseModelPrepPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.prepare_base_model_params_config()
        pre_processed_config = config.pre_processing_params_config()
        prepare_base_model = PrepareBaseModel(config_base_model=prepare_base_model_config, config_preprocessed_data=pre_processed_config)
        prepare_base_model.process_labels()
        # prepare_base_model.process_nums()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = BaseModelPrepPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e
