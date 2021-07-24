import logging
import sys

from logging import getLogger, handlers


def init_logger(name, is_testing):
    logger = getLogger(name)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if not is_testing:
        file_handler = handlers.RotatingFileHandler(f'logs\\{name}.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler.doRollover()
        logger.addHandler(file_handler)
