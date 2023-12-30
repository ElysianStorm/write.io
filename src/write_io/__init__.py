import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers=[
        # File Handler allows to log into a log file (in this case running_logs.log stored in log_filepath)
        logging.FileHandler(log_filepath),
        # Stream Handler allows to log onto the terminal during runtime.
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("write.io_logger")