from write_io.config.configuration import ConfigurationManager
from write_io.components.build_model import BuildModel
from write_io import logger

STAGE_NAME = "Build Model stage"

class BuildModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        build_model_config = config.build_model_config()
        preped_model_config = config.prepare_base_model_params_config()
        model = BuildModel(model_config=build_model_config, config=preped_model_config)
        model.get_model()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = BuildModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e