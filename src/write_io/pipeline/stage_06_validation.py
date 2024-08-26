from write_io.components.prepare_callbacks import PrepareCallback
from write_io.config.configuration import ConfigurationManager
from write_io.components.validation_model import ValidationModelConfig
from write_io import logger

STAGE_NAME = "Validation Model Stage"

class ValidationModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        config = ConfigurationManager()
        prepare_callback_config = config.prepare_callback_config()
        prepare_callback = PrepareCallback(config=prepare_callback_config)
        callback_list = prepare_callback.get_tb_ckpt_callbacks()
        
        validation_model_config = config.validation_model_config()
        model = ValidationModelConfig(validation_model_config=validation_model_config)
        model.get_model()
        model.validate_model(callback_list)

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = ValidationModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e