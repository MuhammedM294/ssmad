"""
run_workflow.py - SSMAD Workflow Execution

"""
__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

import pandas as pd
from pathlib import Path
from ssmad.data_reader import*
from ssmad.anomaly import *
import concurrent.futures
from typing import List , Tuple
from ssmad.utils import create_logger, log_exception, log_time
from functools import partial


# Supported anomaly detection methods
_Detectors = {
        'zscore': Z_Score,
        'smapi-mean': SMAPI , 
        'smapi-median': SMAPI,
        'smdi': SMDI,
        'essmi': ESSMI,
        'smad': SMAD,
        'smds-mean': SMDS,
        'smds-median': SMDS,
        'smci': SMCI,
        'sma_mean':SMCA, 
        'sma_median':SMCA,
    }

logger = create_logger('run_logger')

    

def load_gpis(file):
    """
    Load GPIS from a csv file
    
    paramters:
    ---------
    file: str
        path to the csv file
        
    returns:
    --------
    pd.DataFrame
        a dataframe containing the GPIS
    """
    pointlist = pd.read_csv(file)
    return pointlist


@log_exception(logger)
def load_ts(gpi, ascat_obj , variable = 'sm'):
    """
    Load ASCAT time series for a given gpi
    
    parameters:
    -----------
    gpi: int
        the grid point index
    
    ascat_obj: AscatData
        the ascat object
        
    variable: str
        the variable to load
    
    returns:
    --------
    pd.DataFrame
        a dataframe containing the time series for the given gpi

    """
    ascat_ts = ascat_obj.read(gpi)
    df = pd.DataFrame(ascat_ts.get(variable))
    return df

def validate_date_params(time_step:str,
                         year:Union[int, List[int]] = None,
                         month:Union[int, List[int]] = None,
                         dekad:Union[int, List[int]] = None,
                         week:Union[int, List[int]] = None):
    """
    Validate the date parameters for the anomaly detection
    
    parameters:
    -----------
    
    time_step: str
        the time step of the anomaly detection. Supported time steps are 'month', 'dekad', and 'week'
    
    year: Union[int, List[int]]
        the year of the date or a list of years in case of multiple dates
    
    month: Union[int, List[int]]
        the month of the date or a list of months in case of multiple dates
        
    dekad: Union[int, List[int]]
        the dekad of the date or a list of dekads in case of multiple dates
        
    week: Union[int, List[int]]
        the week of the date or a list of weeks in case of multiple dates
    
    returns:
    --------
    
    dict
        a dictionary containing the date parameters according to the time step
    """
    
    required_params = {'month': ['year', 'month'],
                       'dekad': ['year', 'month', 'dekad'],
                       'week': ['year', 'week']}

    if time_step not in required_params.keys():
        raise ValueError(f"Unsupported time_step: {time_step}. Supported time_steps are {list(required_params.keys())}")

    local_vars = locals()
    missing_params = [param for param in required_params[time_step] if local_vars.get(param) is None ]
    
    if missing_params:
        raise ValueError(f"For time_step '{time_step}', the following parameters must be provided: {', '.join(missing_params)}")

    date_param = {'year': year, 'month': month}

    if time_step == 'dekad':
        
        for dekad in dekad:
            if dekad not in [5, 15, 25]:
                raise ValueError(f"For time_step 'dekad', dekad must be one of 5, 15. or 25"
                                " (corresponding to the first, second, and third dekad of the month)")
        date_param['dekad'] = dekad
    elif time_step == 'week':
        date_param['week'] = week
        date_param.pop('month', None)
        
    # Check if the value lists are of the same length if multiple dates are provided
    
    for _, value in date_param.items():
        if isinstance(value, list):
            if len(value) > 1:
                if len(set([len(value) for value in date_param.values()])) > 1:
                    raise ValueError("The length of the date parameters lists must be the same for multiple dates")
    
        
    return date_param

def anomaly_multiple_dates(gpi:int ,
            method:str= 'zscore',
            variable:str = 'sm',
            time_step:str = 'month',
            year:List[int] = None,
            month:List[int] = None,
            dekad:List[int] = None,
            week:List[int] = None):
    
    # Use the global ascat object
    global ascat_obj
    # Load the time series for the given gpi
    df = load_ts(gpi, ascat_obj)
    # Validate the date parameters
    date_params = validate_date_params(time_step, year, month, dekad, week)
    # Create a list of dictionaries containing the date parameters
    date_params = [dict(zip(date_params.keys(), values)) for values in zip(*date_params.values())]  
    # Define a dictionary to store the results
    results = {}
    
    if method not in _Detectors.keys():
        raise ValueError(f"Anomaly method {method} is not supported. Supported methods are {list(_Detectors.keys())}")
    else:
            
        # Define the anomaly detection parameters
        anomaly_params = {
            'df': df,
            'variable': variable,
            'time_step':time_step,
            }

        # If the method has a metric parameter (e.g. smapi-mean, smapi-median), set the metric parameter
        if '-' in method:
            anomaly_params['metrics'] = [method.split('-')[1]]
            
        for date_param in date_params:
            anomaly = _Detectors[method](**anomaly_params).detect_anomaly(**date_param)
            results[method] = anomaly[method].values[0]




def anomaly_multiple_detectors(
            gpi:int ,
            methods:List[str] = ['zscore','smapi-median'],
            variable:str = 'sm',
            time_step:str = 'month',
            year:int = None,
            month:int = None,
            dekad:int = None,
            week:int = None):
    
    """
    Detect anomalies for a given gpi using single ot multiple anomaly detection methods
    
    parameters:
    -----------
    
    gpi: int
        the grid point index
    
    methods: List[str]
        a list of anomaly detection methods to use. Supported methods are 'zscore', 'smapi-mean', 'smapi-median',
        'smdi', 'essmi', 'smad', 'smds-mean', 'smds-median', 'smci', 'sma_mean', 'sma_median'
        
    variable: str
        the variable to detect the anomaly for. 
    
    time_step: str
        the time step of the anomaly detection. Supported time steps are 'month', 'dekad', and 'week'
        
    year: int
        the year of the date
    
    month: int
        the month of the date
        
    dekad: int
        the dekad of the date
    
    week: int
        the week of the date
    
    returns:
    --------
    
    Tuple[int, dict]
        a tuple containing the grid point index and a dictionary containing the anomalies for the given gpi
    
    """
    
    # Use the global ascat object
    global ascat_obj
    # Load the time series for the given gpi
    df = load_ts(gpi, ascat_obj)
    # Validate the date parameters
    date_params = validate_date_params(time_step, year, month, dekad, week)
    # Define a dictionary to store the results
    results = {}
    
    
    for method in methods:
        
        if method not in _Detectors.keys():
            raise ValueError(f"Anomaly method {method} is not supported. Supported methods are {list(_Detectors.keys())}")
        else:
            
            # Define the anomaly detection parameters
            anomaly_params = {
                'df': df,
                'variable': variable,
                'time_step':time_step,
                }
            
            # If the method has a metric parameter (e.g. smapi-mean, smapi-median), set the metric parameter
            if '-' in method:
                anomaly_params['metrics'] = [method.split('-')[1]]
                
            anomaly = _Detectors[method](**anomaly_params).detect_anomaly(**date_params)
            results[method] = anomaly[method].values[0]
    
    return (gpi, results)


def _finalize(result:Tuple[int, dict] , df:pd.DataFrame , gpis_col = 'point'):
    """
    Store the anomalies in the dataframe according to the grid point index
    
    parameters:
    -----------
    
    result: Tuple[int, dict]
        a tuple containing the grid point index and a dictionary containing the anomalies for the given gpi
    
    df: pd.DataFrame
        the dataframe to store the anomalies in containing the grid point index
    
    gpis_col: str
        the name of the column containing the grid point index
        
    returns:
    --------
    
    pd.DataFrame
        a dataframe containing the anomalies for the given grid point index    
    
    """
    gpi, anomaly = result
    for method, value in anomaly.items():
        df.loc[df[gpis_col] == gpi, method] = value
        
    return df
   
   
@log_time(logger)
def run(gpis_file:str = None, 
        methods:List[str] = ['zscore','smapi-median'],
        variable:str = 'sm',
        time_step:str = 'month',
        year:int = None,
        month:int = None,
        dekad:int = None,
        week:int = None):
    
    """
    Run the anomaly detection workflow for a list of grid point indices using single or multiple anomaly detection methods
    
    parameters:
    -----------
    
    gpis_file: str
        path to the csv file containing the grid point indices
        
    methods: List[str]
        a list of anomaly detection methods to use. Supported methods are 'zscore', 'smapi-mean', 'smapi-median',
        'smdi', 'essmi', 'smad', 'smds-mean', 'smds-median', 'smci', 'sma_mean', 'sma_median'
    
    variable: str
        the variable to detect the anomaly for.
    
    time_step: str
        the time step of the anomaly detection. Supported time steps are 'month', 'dekad', and 'week'
    
    year: int
        the year of the date
    
    month: int
        the month of the date
    
    dekad: int
        the dekad of the date
    
    week: int
        the week of the date
    
    returns:
    --------
    
    pd.DataFrame
        a dataframe containing the anomalies for the given grid point indices
    """
    
    pointlist = load_gpis(gpis_file)
    pointlist = pointlist[:100]
    pre_compute  = partial(anomaly_multiple_detectors,
                           methods=methods,
                           variable=variable,
                           time_step=time_step,
                           year=year,
                           month=month, 
                           dekad = dekad,
                           week = week)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(pre_compute, pointlist['point'])
        for result in results:
            pointlist = _finalize(result, pointlist)
            
        return pointlist
           

if __name__ == "__main__":
    
    # # Example usage
    # Germany = "/home/m294/VSA/Code/pointlist_DEU.csv"
    # ascat_path = Path("/home/m294/VSA/Code/datasets")
    # ascat_obj = AscatData(ascat_path, False)
    
    # test1 = run(gpis_file = Germany,
    #             methods = ['zscore'],
    #             variable = 'sm',
    #             time_step = 'week',
    #             year = 2021,
    #             week = 10)
    # print(test1)
    
    original_dict = {'year': [2022, 2021], 'month': [7, 7]}

    list_of_dicts = [dict(zip(original_dict.keys(), values)) for values in zip(*original_dict.values())]

    list_of_strings = [f"{key}:{value}" for d in list_of_dicts for key, value in d.items()]

    print(list_of_strings)

    
    

  
    
    