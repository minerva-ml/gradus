import logging
import os
import shutil
from pathlib import Path


CACHE_DIRPATH = '.cache'
LOGS_PATH = 'steps.log'


def remove_cache():
    if Path(CACHE_DIRPATH).exists():
        shutil.rmtree(CACHE_DIRPATH)


def remove_logs():
    if Path(LOGS_PATH).exists():
        os.remove(LOGS_PATH)


def prepare_steps_logger():
    print("Redirecting logging to {}.".format(LOGS_PATH))
    remove_logs()
    logger = logging.getLogger('steps')
    for h in logger.handlers:
        logger.removeHandler(h)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(LOGS_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt=message_format)
    logger.addHandler(fh)

