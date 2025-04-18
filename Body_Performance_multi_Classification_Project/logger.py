import logging
import os
from datetime import datetime
import src.constants as const

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"

logs_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Logs")

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    # filename=LOG_FILE_PATH,
    # format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)