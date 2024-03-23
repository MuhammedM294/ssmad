"""
A module for calculating climatology (climate normal) for different time steps (month, dekad, week) based on time series data.
"""

__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

from typing import List, Union
import pandas as pd
from ssmad.preprocess import *



class Aggregator:
    """
    Base class for aggregation

    Attributes:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be aggregated.
        
    variable : str
        The variable/column in the DataFrame to be aggregated.
    
    fillna : bool
        Fill NaN values in the time series data using a moving window average.
        
    fillna_window_size : int
        The size of the moving window for filling NaN values. It is recommended to be an odd number.
    
    smoothing : bool
        Smooth the time series data using a moving window average.
    
    smooth_window_size : int
        The size of the moving window for smoothing(n-days). It is recommended to be an odd number.
        
    timespan : list[str, str] optional
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD', 'YYYY-MM-DD']
    
    resulted_df : pd.DataFrame
        The resulting DataFrame after aggregation.

    Methods:
    --------
    
    _fillna:
        Fills NaN values in the time series data using a moving window average.
    
    _smooth:
        Smooths the time series data using a moving window average.
    
    _set_up_mode():
        Filters the DataFrame based on the parameters provided to perform aggregation on a subset or all of the data.
        
    _filter_df:
        Filters the DataFrame based on specified time/date conditions.
        
    _validate_df_index:
        Validates the input DataFrame type and index.
        
    _validate_variable:
        Validates the variable to be aggregated.
    
    _validate_input:
        Validates the input parameters.
        
    aggregate:
        Aggregates the data based on the specified time step.
        
    """
    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        """
        Initializes the Aggregation class.
            
        """ 
        
        self.original_df = df
        self.var = variable
        self.fillna = fillna
        self.fillna_window_size = fillna_window_size
        self.smoothing = smoothing
        self.smooth_window_size = smooth_window_size
        self.timespan = timespan
        
        self._validate_input()
        self.df = pd.DataFrame(self.original_df[self.var]).resample('D').mean()
        self.df = self.df.truncate(before=timespan[0], after=timespan[1]) if timespan else self.df
        self._fillna()
        self._smooth()
        self.df.dropna(inplace=True)
        self.resulted_df = pd.DataFrame()
    
    def _fillna(self):
        """
        Fills NaN values in the time series data using a moving window average.
        """
        if self.fillna:
            self.df[self.var] = fillna(self.df, self.var, self.fillna_window_size)
            
    def _smooth(self):
        """
        Smooths the time series data using a moving window average.
        """
        if self.smoothing:
            self.df[self.var] = smooth(self.df, self.var, self.smooth_window_size)
    
    
    
    def _validate_df_index(self):
        """
        Validates the input DataFrame type and index.
        
        Raises:
        -------
        
        TypeError:
            If the input DataFrame is not a pandas DataFrame.
        
        ValueError:
            If the input DataFrame index is not a datetime index.
        """
        if not isinstance(self.original_df, pd.DataFrame):
             raise TypeError("df must be a pandas DataFrame")
        
        if not isinstance (self.original_df.index ,pd.DatetimeIndex ):
            raise ValueError("df index must be a datetime index")
     
               
    def _validate_variable(self):
        """
        Validates the variable to be aggregated.
        
        Raises:
        -------
        ValueError:
            If the variable is not found in the input DataFrame columns.
            
        """
        if self.var not in self.original_df.columns:
            raise ValueError(f"Variable '{self.var}' not found in the input DataFrame columns.")
        
    def _validate_fillna_smoothing(self):
   
        """
        Validates the smoothing parameters.

        Raises:
        -------
        ValueError:
            - If the window size is not provided when smoothing is enabled.
        TypeError:
            - If the smoothing parameter is not a boolean value.
            - If the window size parameter is not an integer value when smoothing is enabled.
        """
        
        if any([self.fillna and self.fillna_window_size is None,\
            self.smoothing and self.smooth_window_size is None]):
            
            raise ValueError("window size must be provided when 'fillna' or 'smoothing' is enabled")

        
    def _validate_input(self):
        """
        Validates the input parameters.
        """
        self._validate_df_index()
        self._validate_variable()
        self._validate_fillna_smoothing()
    

    def aggregate(self , **kwargs):
        """
        Aggregates the data based on the specified .
        """
        return filter_df(self.df , **kwargs)


class MonthlyAggregator(Aggregator):
    """
    Aggregates the time series data based on month-based time step.     
    """

    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = 3,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)

    def aggregate(self, **kwargs):
        
        self.resulted_df[f"{self.var}-avg"] = monthly_agg(self.df, self.var)

        return filter_df(self.resulted_df , **kwargs)

class DekadalAggregator(Aggregator):
    """
    Aggregates the data based on dekad-based time step.
    """

    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)

    def aggregate(self , **kwargs):
        
        self.resulted_df[f"{self.var}-avg"]  = dekadal_agg(self.df, self.var)
        
        return filter_df(self.resulted_df , **kwargs)

class WeeklyAggregator(Aggregator):
    """
    Aggregates the time series data based on week-based time step.
    """

    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)

    def aggregate(self , **kwargs):
        
        self.resulted_df[f"{self.var}-avg"] = weekly_agg(self.df, self.var)
        
        return filter_df(self.resulted_df , **kwargs)
    
class BimonthlyAggregator(Aggregator):
    """
    Aggregates the time series data based on bimonthly (twice a month) time step.
    """

    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)
        

    def aggregate(self , **kwargs):
        
        self.resulted_df[f"{self.var}-avg"] = bimonthly_agg(self.df, self.var)
        
        return filter_df(self.resulted_df , **kwargs)
    
class DailyAggregator(Aggregator):
    """
    Aggregates the time series data based on daily time step.
    """

    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None):
        
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)
        

    def aggregate(self , **kwargs):
        self.resulted_df[f"{self.var}-avg"] = self.df[self.var]
        return filter_df(self.resulted_df.drop_duplicates() , **kwargs)
        


class Climatology(Aggregator):
    """
    A class for calculating climatology(climate normal) for time series data.

    Attributes:
    ----------
    df_original: pd.DataFrame
        The original input DataFrame before resampling and removing NaN values.

    df: pd.DataFrame
        The input DataFrame containing the preprocessed data to be aggregated.

    variable: str
        The variable/column in the DataFrame to be aggregated.
        
    fillna: bool
        Fill NaN values in the time series data using a moving window average.
        
    fillna_window_size: int
        The size of the moving window for filling NaN values. It is recommended to be an odd number.
        
    smoothing: bool
        Smooth the time series data using a moving window average.
    
    smooth_window_size: int
        The size of the moving window for smoothing(n-days). It is recommended to be an odd number.
    
    timespan: list[str, str] optional
        The start and end dates for a timespan to be aggregated. Format: ['YYYY-MM-DD ]

    time_step: str
        The time step for aggregation. Supported values: 'day', 'week', 'dekad', 'bimonth', 'month'.

    normal_metrics: List[str]
        The metrics to be used in the climatology computation. Supported values: 'mean', 'median', 'min', 'max', etc.

    clima_df: pd.DataFrame
        The DataFrame containing climatology information.

    Methods:
    -------
    aggregate:
        Aggregates the data based on the time step and metrics provided.
    
    _validate_time_step:
        Validates the time step.
        
    _validate_metrics:
        Validates the metrics to be used in the climatology computation.

    compute_normals:
        Calculates climatology based on the aggregated data.
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, 
                 fillna:bool = False,fillna_window_size:int = None,
                 smoothing = False, smooth_window_size = None,
                 timespan: List[str] = None , 
                 time_step: str = "month", normal_metrics: List[str] = ["mean"] ):
        """
        Initializes the Climatology class.
        """
        self.time_step = time_step
        self.normal_metrics = normal_metrics
        self.valid_time_steps = ["month", "dekad", "week", "day", "bimonth"]
        self.valid_metrics = ["mean", "median", "min", "max"]
        super().__init__(df, variable, fillna,fillna_window_size, smoothing, smooth_window_size, timespan)
        self.clim_df = pd.DataFrame()
        
    
      
    def _validate_time_step(self,) -> None:
        """
        Validates the time step.
        
        Raises:
        -------
        ValueError:
            If the time step is not one of the supported values.
            
        """
        if self.time_step not in self.valid_time_steps:
            raise ValueError(f"Invalid time step '{self.time_step}'. Supported values: {self.valid_time_steps}.")
        
        
    def _validate_metrics(self):
        """
        Validates the metrics to be used in the climatology computation.

        Raises:
            ValueError: If the metric is not one of the supported values.
            
        """
        for metric in self.normal_metrics:
            if metric not in self.valid_metrics:
                raise ValueError(f"Invalid metric '{metric}'. Supported values: {self.valid_metrics}.")
            
    def _validate_input(self):
        super()._validate_input()
        self._validate_time_step()
        self._validate_metrics()
          
    
    def aggregate(self):
        """
        Aggregates the data based on the time step and metrics provided.
        
        """
        if self.time_step == 'month':
             return MonthlyAggregator(self.df, self.var ,self.fillna,
                                        self.fillna_window_size, self.smoothing, self.smooth_window_size, self.timespan).aggregate()
             
        elif self.time_step == 'week':
            return WeeklyAggregator(self.df, self.var ,self.fillna,
                                        self.fillna_window_size, self.smoothing, self.smooth_window_size, self.timespan).aggregate()
            
        elif self.time_step == 'dekad':
            return DekadalAggregator(self.df, self.var ,self.fillna,
                                        self.fillna_window_size, self.smoothing, self.smooth_window_size, self.timespan).aggregate()
            
        elif self.time_step == 'bimonth':
            return BimonthlyAggregator(self.df, self.var ,self.fillna,
                                        self.fillna_window_size, self.smoothing, self.smooth_window_size, self.timespan).aggregate()
            
        elif self.time_step == 'day':
            return DailyAggregator(self.df, self.var ,self.fillna,
                                        self.fillna_window_size, self.smoothing, self.smooth_window_size, self.timespan).aggregate()
         

    def compute_normals(self, **kwargs) -> pd.DataFrame:
        """
        Calculates climatology based on the aggregated data.

        Parameters:
        -----------
        kwargs:
            Additional time/date filtering parameters.

        Returns:
        --------
        pd.DataFrame
            The DataFrame containing climatology information.
        """
        self.clim_df = compute_clim(self.aggregate(), self.time_step, f"{self.var}-avg", self.normal_metrics)         
        
        return filter_df(self.clim_df, **kwargs)
    

if __name__ == "__main__":
    
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Morocco
    lat = 33.201
    lon = -7.373
        
    sm_ts = extract_obs_ts((lon, lat), ascat_path, obs_type="sm" , read_bulk=False)
    cl = Climatology(sm_ts, "sm", time_step="month", normal_metrics=["mean",'median'] , timespan=['2007-01-01', '2009-12-31'])
    print(cl.compute_normals())
   