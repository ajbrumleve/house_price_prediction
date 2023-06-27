import logging
import timeit
from datetime import datetime

def log(func):
    """Decorator that logs the execution of a function.

        This decorator logs the start and end of a function's execution, along with the elapsed time.

        Args:
            func: The function to be decorated.

        Returns:
            The decorated function.

        Example:
            @log
            def my_function():
                # Function implementation

            my_function()  # The function will be logged when executed
    """
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
    """Configure logging formatter for a logger.

        This function sets up a logging formatter for a logger with a specified name.
        It configures a file handler and a specific log message format.

        Args:
            logger: The logger to configure the formatter for.
            name (str): The name of the logger.

        Returns:
            None

        Example:
            logger = logging.getLogger("wxApp")
            logging_formatter(logger, "wxApp")
    """
    fh = logging.FileHandler('logs/streamlit-log.log')
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)