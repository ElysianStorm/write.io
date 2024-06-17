from write_io.config.configuration import ConfigurationManager
from write_io.components.prepare_model import PrepareModel
from write_io import logger

STAGE_NAME = "Prepare Model stage"

class ModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.prepare_model_params_config()
        prepare_base_model = PrepareModel(config_model=prepare_model_config)
        # prepare_base_model.process_nums()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = ModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e
