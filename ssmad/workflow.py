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
from ssmad.utils import create_logger, log_exception, log_time , load_gpis_by_country


# Supported anomaly detection methods
_Detectors = {
    'zscore': ZScore,
    'smapi-mean': SMAPI,
    'smapi-median': SMAPI,
    'smdi': SMDI,
    'smca-mean': SMCA,
    'smca-median': SMCA,
    'smad': SMAD,
    'smci': SMCI, 
    'smds': SMDS,
    'essmi': ESSMI,
    'beta':ParaDis, 
    'gamma':ParaDis,
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
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan
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
    
    """
    Validate the date parameters for the anomaly detection workflow.         
    """
    
    
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
            fillna:bool = False,
            fillna_window_size:int = None,
            smoothing:bool = False,
            smooth_window_size:int = None,
            year:Union[int, List[int]] = None,
            month:Union[int, List[int]] = None,
            dekad:Union[int, List[int]]  = None,
            week:Union[int, List[int]]  = None , 
            bimonth: Union[int, List[int]] = None , 
            day: Union[int,List[int]]= None, 
            timespan: List[str] = None)-> Tuple[int, Dict[str, float]]:
    
    """
    The anomaly detection workflow for a given grid point index (GPI).
    
    parameters:
    -----------
    
    gpi: int
        The grid point index
        
    methods: str or list of str
        The anomaly detection methods to be used. Supported methods are: 'zscore', 'smapi-mean', 'smapi-median',
        'smdi', 'smca-mean', 'smca-median', 'smad', 'smci', 'smds', 'essmi', 'beta', 'gamma'
        
    variable: str
        The variable to be used for the anomaly detection. Supported values: 'sm'
        
    time_step: str
        The time step of the date parameters. Supported values: 'month', 'dekad', 'week', 'bimonth', 'day'
        
    fillna: bool
        Whether to fill missing values in the time series
        
    fillna_window_size: int
        The window size for the fillna method
        
    smoothing: bool
        Whether to apply smoothing to the time series
        
    smooth_window_size: int
        The window size for the smoothing method
        
    year: int or list of ints
        The year(s) for the date parameters
        
    month: int or list of ints
        The month(s) for the date parameters
        
    dekad: int or list of ints
        The dekad(s) for the date parameters
        
    week: int or list of ints
        The week(s) for the date parameters
        
    bimonth: int or list of ints
        The bimonth(s) for the date parameters
        
    day: int or list of ints
        The day(s) for the date parameters
        
    timespan: list of str
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ] 
        example: ['2012-01-01', '2012-12-31']
        
    returns:
    --------
    Tuple[int, Dict[str, float]]
        The grid point index and the results of the anomaly detection methods
        
    
    Example:
    --------
    Single anomaly detection method with a single date parameter:
    
        Compute the anomaly using the 'zscore' method based on monthly climate normal 
        for the grid point index 4841504 for the month of December 2012:
        
            anomaly_worlflow(4841504, methods=['zscore'], variable='sm', time_step='month', year=2012, month=12)
        
        
        
    Multiple anomaly detection methods with multiple date parameters:
    
       Compute the anomalies using the 'beta' and 'gamma' methods based on weekly climate normals
       for the grid point index 4841504 for the months of December 2012 and November 2011:
       
           anomaly_worlflow(4841504, methods=['beta','gamma'], variable='sm', time_step='week', year=[2012,2011], month=[12,11])
    """
    
    # Use the global ascat object
    global ascat_obj
    # Load the time series for the given gpi
    # df = extract_obs_ts(gpi, ascat_path, obs_type="sm" , read_bulk=False)
    df = load_ts(gpi, ascat_obj)
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
                'fillna': fillna,
                'fillna_window_size': fillna_window_size,
                'smoothing': smoothing,
                'smooth_window_size': smooth_window_size,
                'timespan': timespan,
                }

            # If the method has a metric parameter (e.g. smapi-mean, smapi-median), set the metric parameter
            if '-' in method:   
                anomaly_params['normal_metrics'] = [method.split('-')[1]]
                
            elif method in ['beta','gamma']:
                anomaly_params['dist'] = [method]
                
            for date_param in date_params:
                anomaly = _Detectors[method](**anomaly_params).detect_anomaly(**date_param)
                date_str = f"-".join(str(value) for value in date_param.values())
                results[method+ f"({date_str})"] = anomaly[method].values[0]
        
    return (gpi , results)




def _finalize(result:Tuple[int, dict] , df:pd.DataFrame , gpis_col = 'point'):

    gpi, anomaly = result
    for method, value in anomaly.items():
        df.loc[df[gpis_col] == gpi, method] = value
        
    return df


   
@log_time(logger)
def run(
        country:str = None, 
        methods:Union[str , List[str]] = ['zscore'],
        variable:str = 'sm',
        time_step:str = 'month',
        fillna:bool = False,
        fillna_window_size:int = None,
        smoothing:bool = False,
        smooth_window_size:int = None,
        timespan: List[str] = None,
        year:List[int]= None,
        month:List[int] = None,
        dekad:List[int] = None,
        week:List[int] = None, 
        bimonth: List[int] = None,
        day: List[int] = None,
        workers:int = None) -> pd.DataFrame:
    
    """
    Run the anomaly detection workflow for multiple grid point indices with multiprocessing support.
    """
    
    
    pointlist = load_gpis_by_country(country)
    pointlist = pointlist[:10]
    pre_compute  = partial(anomaly_worlflow,
                           methods=methods,
                           variable=variable,
                           time_step=time_step,
                           fillna=fillna,
                           fillna_window_size=fillna_window_size,
                           smoothing=smoothing,
                           smooth_window_size=smooth_window_size,
                           year=year,
                           month=month, 
                           dekad = dekad,
                           week = week, 
                           bimonth = bimonth,
                           day = day,
                           timespan = timespan)
    
    
    with ProcessPoolExecutor(workers) as executor:
        results = executor.map(pre_compute, pointlist['point'])
        for result in results:
            pointlist = _finalize(result, pointlist)
            
        return pointlist
    


if __name__ == "__main__":
    
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    # # Example usage
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    ascat_obj = AscatData(ascat_path, False)
    df = run('germany', methods=['zscore'], variable='sm', time_step='month', year=2012, month=12 ,
             fillna=True, fillna_window_size=3, smoothing=True, smooth_window_size=31,
             )
    
    print(df)
    
    
    
    
    

   
    
   