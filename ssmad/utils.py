import logging
from functools import wraps
from time import time
import requests
from io import StringIO
import pycountry
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


def get_country_code(country_name):
    """
    Get the ISO 3166-1 alpha-3 country code for a given country name.
    
    parameters:
    ----------
    
    country_name: str
        name of the country
        
    returns:
    -------
    
    country_code: str
        ISO 3166-1 alpha-3 country code
    """
    try:
        # Get ISO alpha-3 code for the provided country name
        country = pycountry.countries.lookup(country_name)
        return country.alpha_3
    except LookupError:
        print("Country name not found.")
        return None

def load_gpis_by_country(country , grid = "fibgrid_n6600000" , format = "csv"):
    
    """
    Load the GPIS based on the country name from the DGG API 
    Source: https://dgg.geo.tuwien.ac.at/
    
    parameters:
    ----------
    country: str
        name of the country
        
    grid: str
        name of the grid to be used. Default is "fibgrid_n6600000". Supported grids are:
        - fibgrid_n6600000 (Fibonacci 6.5 km)
        - fibgrid_n1650000 (Fibonacci 12.5 km)
        - fibgrid_n430000  (Fibonacci 25 km)
        - warp (WARP)
        
    format: str
        format of the data to be returned. Default is "csv". Supported formats are:
        - csv
        - json
    """
    
    country = get_country_code(country)
    
    # Construct the URL based on the provided country name, grid, and format
    url = f"https://dgg.geo.tuwien.ac.at/get_points/?grid={grid}&country={country.upper()}&format=csv"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            csv_data = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            return df
        else:
            print(f"Failed to download CSV file. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


    
if __name__ == "__main__":
    print(pycountry.countries.lookup("hahah"))
    