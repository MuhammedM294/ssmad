"""
run_workflow.py - SSMAD Workflow Execution

"""
__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

from typing import List, Tuple, Union, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
from pathlib import Path
from ssmad.data_reader import *
from ssmad.anomaly import *
from ssmad.utils import create_logger, log_exception, log_time


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


def _validate_param(param_name: str, param_value: List[int]) -> None:
    """
    Validate a date parameter to ensure it's not None and is a list of integers.
    
    parameters:
    -----------
    param_name: str
        the name of the parameter
    
    param_value: List[int]
        the value of the parameter
        
    Raises:
    -------
    
    ValueError
        if the parameter is None
        if the parameter is not a list
        if the parameter contains non-integer values
    
    """
    if param_value is None:
        raise ValueError(f"The '{param_name}' parameter must be provided")
    if not isinstance(param_value, list):
        raise ValueError(f"The '{param_name}' parameter must be a list")
    if not all(isinstance(item, int) for item in param_value):
        raise ValueError(f"All items in the '{param_name}' parameter must be integers")


def _validate_dekad(dekad: List[int]) -> None:
    """
    Validate dekad values to ensure they are one of 5, 15, or 25.
    
    parameters:
    -----------
    
    dekad: List[int]
        a list of dekad values
        
    Raises:
    -------
    
    ValueError
        if any of the dekad values is not one of 5, 15, or 25
    """
    if not all(d in [5, 15, 25] for d in dekad):
        raise ValueError("For time_step 'dekad', dekad must be one of 5, 15, or 25"
                         " for each month, 5 first dekad, 15 second dekad, 25 third dekad.")

def _validate_time_step(time_step: str, required_params: Dict[str, List[str]]) -> None:
    """
    Validate the provided time step. Supported time steps are month, dekad, and week.
    
    parameters:
    -----------
    
    time_step: str
        the time step to validate
    
    required_params: Dict[str, List[str]]
        a dictionary containing the required parameters for each time step
    
    Raises:
    -------
    
    ValueError
        if the time step is not supported
        
    """
    if time_step not in required_params:
        raise ValueError(f"Unsupported time_step: {time_step}. "
                         f"Supported time_steps are {list(required_params.keys())}")

def _validate_required_params(time_step: str, required_params: Dict[str, List[str]], local_vars: Dict[str, List[int]]) -> None:
    """
    Check if all required date parameters for a given time step are provided.
    
    parameters:
    -----------
    
    time_step: str
        the time step to validate
        
    required_params: Dict[str, List[str]]
        a dictionary containing the required parameters for each time step
        
    local_vars: Dict[str, List[int]]
        a dictionary containing the local variables and their values
    
    Raises:
    -------
    
    ValueError
        if any of the required parameters is not provided
        
    Examples:
    ---------
    
    For time_step 'dekad', the required parameters are year, month, and dekad.
    For time_step 'week', the required parameters are year and week.
    
    """
    missing_params = [param for param in required_params[time_step] if local_vars.get(param) is None]
    if missing_params:
        raise ValueError(f"For time_step '{time_step}', the following parameters must be provided: "
                         f"{', '.join(missing_params)}")

def validate_date_params(time_step: str,
                         year: List[int] = None,
                         month: List[int] = None,
                         dekad: List[int] = None,
                         week: List[int] = None) -> Dict[str, List[int]]:
    """
    Validate the date parameters of the workflow.
    
    parameters:
    -----------
    
    time_step: str
        the time step to validate
        
    year: List[int]
        a list of years
        
    month: List[int]
        a list of months
        
    dekad: List[int]
        a list of dekads
        
    week: List[int]
        a list of weeks
        
    Raises:
    -------
    
    ValueError
        if the length of the date parameters lists are not the same
        if the time step is not supported
        if any of the required parameters is not provided
        
    Examples:
    ---------
    
    For a single date:
          year = [2022] , month = [10] , dekad = [5] corresponds to the first dekad of October 2022.
    For multiple dates:
          year = [2022, 2021, 2020] , month = [10, 11, 5] , dekad = [15, 25, 5] 
          corresponds to the second dekad of October 2022, the third dekad of November 2021, and the first dekad of May 2020.
    
    """
    date_param = {'year': year, 'month': month}

    # Validating dekad or week based on time_step
    if time_step == 'dekad':
        _validate_dekad(dekad)
        date_param['dekad'] = dekad
    elif time_step == 'week':
        date_param['week'] = week
        date_param.pop('month', None)
        
    # Validation for parameters
    for param_name, param_value in date_param.items():
        _validate_param(param_name, param_value)

    # Checking if the value lists are of the same length
    if len(set(map(len, date_param.values()))) > 1:
        raise ValueError("The length of the date parameters lists must be the same for multiple dates")

    # Checking if required parameters are provided
    required_params = {'month': ['year', 'month'],
                       'dekad': ['year', 'month', 'dekad'],
                       'week': ['year', 'week']}

    local_vars = locals()
    _validate_time_step(time_step, required_params)
    _validate_required_params(time_step, required_params, local_vars)

    return date_param


def anomaly_worlflow(gpi:int ,
            methods:str= ['zscore'],
            variable:str = 'sm',
            time_step:str = 'month',
            year:List[int] = None,
            month:List[int] = None,
            dekad:List[int] = None,
            week:List[int] = None)-> Tuple[int, Dict[str, float]]:
    
    """
    Run the anomaly detection workflow for a given grid point index.
    
    parameters:
    -----------
    
    gpi: int
        the grid point index
    
    methods: Union[str, List[str]]
        the anomaly detection methods to use. Supported methods are one of the following:
        'zscore', 'smapi-mean', 'smapi-median', 'smdi', 'essmi', 'smad', 'smds-mean', 'smds-median', 'smci', 'sma_mean', 'sma_median'
    
    variable: str
        the variable to use for anomaly detection. The column name in the provided time series dataframe
        
    time_step: str
        the time step to use for anomaly detection. Supported time steps are 'month', 'dekad', and 'week'
        
    year: Union[int, List[int]]
        a single year or multiple years in case of multiple dates
    
    month: Union[int, List[int]]
        a single month or multiple months in case of multiple dates
        
    dekad: Union[int, List[int]]
        a single dekad or multiple dekads in case of multiple dates
    
    week: Union[int, List[int]]
        a single week or multiple weeks in case of multiple dates
        
    returns:
    --------
    
    Tuple[int, Dict[str, float]]
        a tuple containing the grid point index and a dictionary containing the anomalies for the given gpi
        
    Examples:
    ---------
    
    For a single date with single method:
            anomaly_workflow(gpi=12345, methods=['zscore'], variable='sm', time_step='dekad', year=2022, month=10, dekad=5)
            computes the z-score anomaly for the first dekad of October 2022 for the given gpi.
            
    for a single date with multiple methods:
            anomaly_workflow(gpi=12345, methods=['zscore', 'smapi-median'], variable='sm', time_step='dekad', year=2022, month=10, dekad=5)
            computes the z-score and smapi-median anomalies for the first dekad of October 2022 for the given gpi.
            
    For multiple dates with multiple methods
            anomaly_workflow(gpi=12345, methods=['zscore', 'smapi-median'], variable='sm', time_step='dekad', year=[2022,2022], month=[10,10], dekad=[5,15])
            computes the z-score and smapi-median anomalies for the first and second dekads of October 2022 for the given gpi.
    """
    
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
    for method in methods:
        if method not in _Detectors.keys():
            raise ValueError(f"Anomaly method '{method}' is not supported."
                              f"Supported methods are one of the following: {tuple(_Detectors.keys())}")
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
                date_str = f"-".join(str(value) for value in date_param.values())
                results[method+"_"+date_str] = anomaly[method].values[0]
        
    return (gpi , results)





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
def run(
        gpis_file_path:str = None, 
        methods:Union[str , List[str]] = ['zscore','smapi-median'],
        variable:str = 'sm',
        time_step:str = 'month',
        year:List[int]= None,
        month:List[int] = None,
        dekad:List[int] = None,
        week:List[int] = None):
    
    """
    Multiprocessing workflow execution for anomaly detection. Decorated with a logger to log the time taken for the workflow to execute.
    
    Returns:
    --------
    
    pd.DataFrame
        a dataframe containing the anomalies for each grid point index.
        
    Examples:
    ---------
    
    For a single date with single method:
            run(gpis_file_path='pointlist_DEU.csv', methods=['zscore'], variable='sm', time_step='dekad', year=2022, month=10, dekad=5)
            computes the z-score anomaly for the first dekad of October 2022 for the given gpis.
            
    for a single date with multiple methods:
            run(gpis_file_path='pointlist_DEU.csv', methods=['zscore', 'smapi-median'], variable='sm', time_step='dekad', year=2022, month=10, dekad=5)
            computes the z-score and smapi-median anomalies for the first dekad of October 2022 for the given gpis.
            
    For multiple dates with multiple methods
            run(gpis_file_path='pointlist_DEU.csv', methods=['zscore', 'smapi-median'], variable='sm', time_step='dekad', year=[2022,2022], month=[10,10], dekad=[5,15])
            computes the z-score and smapi-median anomalies for the first and second dekads of October 2022 for the given gpis.
    """
 
    
    pointlist = load_gpis(gpis_file_path)
    pointlist = pointlist[:10]
    pre_compute  = partial(anomaly_worlflow,
                           methods=methods,
                           variable=variable,
                           time_step=time_step,
                           year=year,
                           month=month, 
                           dekad = dekad,
                           week = week)
    with ProcessPoolExecutor() as executor:
        results = executor.map(pre_compute, pointlist['point'])
        for result in results:
            pointlist = _finalize(result, pointlist)
            
        return pointlist
    



if __name__ == "__main__":
    
    # # Example usage
    Germany = "/home/m294/VSA/Code/pointlist_DEU.csv"
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    ascat_obj = AscatData(ascat_path, False)
  
    
    df = run(Germany, methods =['zscore'], variable = 'sm', time_step = 'dekad', year = [2022,2022], month=[10,10], dekad=[5,15])
    
    print(df)
    
    