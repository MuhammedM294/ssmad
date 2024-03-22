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
from ssmad.anomaly_detectors import *
from ssmad.utils import create_logger, log_exception, log_time


# Supported anomaly detection methods
_Detectors = {
    'zscore': ZScore,
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
 
    if param_value is None:
        raise ValueError(f"The '{param_name}' parameter must be provided")
    if not (isinstance(param_value, (list,int))):
        raise ValueError(f"The '{param_name}' parameter must be an int of list of ints")
   


def _validate_required_params(time_step: str, required_params: Dict[str, List[str]], local_vars: Dict[str, List[int]]) -> None:
 
    missing_params = [param for param in required_params[time_step] if local_vars.get(param) is None]
    if missing_params:
        raise ValueError(f"For time_step '{time_step}', the following parameters must be provided: "
                         f"{', '.join(missing_params)}")

def validate_date_params(time_step: str,
                         year: Union[int, List[int]]= None,
                         month: Union[int, List[int]] = None,
                         dekad: Union[int, List[int]] = None,
                         week: Union[int, List[int]] = None,
                         bimonth: Union[int, List[int]] = None, 
                         day: Union[int, List[int]] = None) -> Dict[str, List[int]]:
    
    
    
    year = [year] if isinstance(year, int) else year
    month = [month] if isinstance(month, int) else month
    dekad = [dekad] if isinstance(dekad, int) else dekad
    week = [week] if isinstance(week, int) else week
    bimonth = [bimonth] if isinstance(bimonth, int) else bimonth
    day = [day] if isinstance(day, int) else day
    

    date_param = {'year': year, 'month': month}

    # Validating dekad or week based on time_step
    
    if time_step == 'month':
        pass
    elif time_step == 'dekad':
        date_param['dekad'] = dekad
    elif time_step == 'bimonth':
        date_param['bimonth'] = bimonth
        
    elif time_step == 'day':
        date_param['day'] = day
        
    elif time_step == 'week':
        date_param['week'] = week
        date_param.pop('month')
        
    else:
        raise ValueError(f"Unsupported time_step: {time_step}. Supported time_steps are month, dekad, week, bimonth, day")
        
        
    # Validation for parameters
    for param_name, param_value in date_param.items():
        _validate_param(param_name, param_value)

    # Checking if the value lists are of the same length
    if len(set(map(len, date_param.values()))) > 1:
        raise ValueError("The length of the date parameters lists must be the same for multiple dates")

    # Checking if required parameters are provided
    required_params = {'month': ['year', 'month'],
                       'dekad': ['year', 'month', 'dekad'],
                       'week': ['year', 'week'], 
                       'bimonth': ['year', 'month','bimonth'],
                       'day': ['year', 'month', 'day']}

    local_vars = locals()
    _validate_required_params(time_step, required_params, local_vars)

    return date_param


def anomaly_worlflow(gpi:int ,
            methods:str= ['zscore'],
            variable:str = 'sm',
            time_step:str = 'month',
            year:Union[int, List[int]] = None,
            month:Union[int, List[int]] = None,
            dekad:Union[int, List[int]]  = None,
            week:Union[int, List[int]]  = None , 
            bimonth: Union[int, List[int]] = None , 
            day: Union[int,List[int]]= None)-> Tuple[int, Dict[str, float]]:
    

    
    # Use the global ascat object
    global ascat_obj
    # Load the time series for the given gpi
    df = extract_obs_ts(gpi, ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    # df = load_ts(gpi, ascat_obj)
    # Validate the date parameters
    date_params = validate_date_params(time_step, year, month, dekad, week, bimonth, day)
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
                if method.split('-')[1]  == 'dis':
                    anomaly_params['dis'] = method.split('-')[0]
                    
                else:
                    anomaly_params['normal_metrics'] = [method.split('-')[1]]
                
            for date_param in date_params:
                anomaly = _Detectors[method](**anomaly_params).detect_anomaly(**date_param)
                date_str = f"-".join(str(value) for value in date_param.values())
                results[method+"_"+date_str] = anomaly[method].values[0]
        
    return (gpi , results , df)




def _finalize(result:Tuple[int, dict] , df:pd.DataFrame , gpis_col = 'point'):

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
    
    
    pointlist = load_gpis(gpis_file_path)
    pointlist = pointlist[10:20]
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
    
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    # # Example usage
    Germany = "/home/m294/VSA/Code/pointlist_DEU.csv"
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    ascat_obj = AscatData(ascat_path, False)
  
    
    # df = run(Germany, methods =['zscore'], 
    #          variable = 'sm', time_step = 'month',
    #          year = 2011, month=11)
    
    
    
        
    po = 4854801

        
    sm_ts = extract_obs_ts(po, ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    x = anomaly_worlflow(po, methods =['zscore'], 
             variable = 'sm', time_step = 'month',
             year = 2011, month=11)
    
    print(x)
    
   