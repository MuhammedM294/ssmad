import logging
from functools import wraps
from time import time
import pandas as pd
import numpy as np


def create_logger(name, level=logging.DEBUG):
    """
    Create a logger with the given name and level
    
    parameters:
    ----------
    
    name: str
        name of the logger
    level: logging.LEVEL
        level of the logger
    
    returns:
    -------
    logger: logging.logger
        a logger object
    """
    
    #create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    #create a file to store all the logs exceptions
    logfile = logging.FileHandler('run_logger.log')
    
    #create a formatter and set the formatter for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    
    return logger


def log_exception(logger):
    """
    A decorator to log exceptions in a function
    
    parameters:
    ----------
    logger: logging.logger
        a logger object
        
    returns:
    -------
    decorator: function
        a decorator function
    
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                issue = f"Exception '{e}'in "+  func.__name__ + "\n"
                issue += "=============================\n"
                logger.exception(issue)
               
        return wrapper
    return decorator
    
    
def log_time(logger):
    """
    A decorator to log the time taken by a function
    
    parameters:
    ----------
    logger: logging.logger
        a logger object
        
    returns:
    -------
    decorator: function
        a decorator function
    
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.info("Function %s took %s seconds", func.__name__, end-start)
            return result
        return wrapper
    return decorator



    