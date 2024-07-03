from write_io.config.configuration import ConfigurationManager
from write_io.components.training_model import TrainingModel
from write_io import logger

STAGE_NAME = "Training Model Stage"

class TrainingModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_model_config = config.training_model_config()
        model = TrainingModel(model_config=training_model_config)
        model.train_model()

if __name__ == '__main__':
    try:
        logger.info(f"------------- STAGE {STAGE_NAME} Started -------------")
        obj = TrainingModelPipeline()
        obj.main()
        logger.info(f"+++++++++++++ STAGE {STAGE_NAME} Completed +++++++++++++")
    except Exception as e:
        logger.exception(e)
        raise e