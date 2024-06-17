from write_io.config.configuration import ConfigurationManager
from write_io.components.train_model import TrainModel
from write_io import logger

STAGE_NAME = "Train Model stage"

class TrainModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        # prepare_model_config = config.prepare_model_params_config()
        train_model = TrainModel(config_model=prepare_model_config)
        # prepare_base_model = TrainModel()
        # prepare_base_model.process_nums()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e
