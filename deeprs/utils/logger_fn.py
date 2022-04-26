# @Time  : 2022/4/25 20:55
# @Author: xizhong
# @Desc  :

import logging


def set_logger(level=logging.INFO, log_file='../logs/model.log'):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_logging = logging.FileHandler(log_file)
    file_logging.setLevel(level)
    file_logging.setFormatter(formatter)
    logger.addHandler(file_logging)
    return logger
