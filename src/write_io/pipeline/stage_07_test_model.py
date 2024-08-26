from write_io.config.configuration import ConfigurationManager
from write_io.components.test_model import TestModelConfig
from write_io import logger

STAGE_NAME = "Test Model Stage"

class TestModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        
        test_model_config = config.test_model_config()
        model = TestModelConfig(test_model_config=test_model_config)
        model.get_model()
        model.test_model()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = TestModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e