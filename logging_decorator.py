import logging
import timeit
from datetime import datetime

def log(func):
    def wrapper(*args,**kwargs):
        t_section = timeit.default_timer()
        name = func.__name__
        logger = logging.getLogger("wxApp")
        logger.setLevel(logging.INFO)
        logging_formatter(logger,name)
        logger.info(f"Running function: {name}")
        result = func(*args,**kwargs)
        logger.info(f"{name} finished in {timeit.default_timer() - t_section} seconds")
        return result

    return wrapper

def logging_formatter(logger, name):
    fh = logging.FileHandler('logs/test-log.log')
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)