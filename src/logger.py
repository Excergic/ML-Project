import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_&M_%S')}.log"

logs_path = os.path.join(os.getcwd(),"logs")#--> ''' Specifies the path where the log files will be stored, 
#The logs directory is created within the current working directory (os.getcwd()).'''

os.makedirs(logs_path, exist_ok=True) #-->this creates log directory if ti des not exists,
#'exist_ok = True, prevents error if directory logs is already exists.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) #--> Full path to the log file, combining log_path, LOG_FILE

logging.basicConfig(
    filename= LOG_FILE_PATH,# all log messages will directed to this file
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO, # Confirmatiohn that thing are working expexted
    # There is also level = logging.DEBUG--> used to detailed Information, typically of interest only,
    #-when diagnosing problem.
)

