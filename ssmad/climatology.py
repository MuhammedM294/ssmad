"""
A module for calculating climatology for different time steps (month, dekad, week) 
using different metrics (mean, median, min, max).
"""

__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

import pandas as pd
from typing import List, Union

class Aggregator:
    """
    Base class for aggregation

    Attributes:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be aggregated.
        
    variable : str
        The variable/column in the DataFrame to be aggregated.
        
    time_step : str
        The time step for aggregation. Supported values: 'month', 'dekad', 'week'.
        
    metrics : List[str]
        The list of metrics to be applied during aggregation. Supported values: 'mean', 'median', 'min', 'max'.
        
    mode : str
        Perform aggregation on the entire dataset or on a subset of the dataset. Supported values: 'all', 'subset'.
        
    start_date : str
        The start date for the aggregation in case of subset mode. The format should be 'YYYY-MM-DD'.
    end_date : str
        The end date for the aggregation in case of subset mode. The format should be 'YYYY-MM-DD'.
        
    resulted_df : pd.DataFrame
        The resulting DataFrame after aggregation.

    Methods:
    --------
    aggregate():
        Aggregates the data based on the specified time step and metrics.
    
    _validate_time_step():
        Validates the time step.
    
    _validate_variable():
        Validates the variable.
        
    _validate_input():
        Validates the input parameters.
    
    _set_up_mode():
        Filters the DataFrame based on the parameters provided to perform aggregation on a subset or all of the data.
    """
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str , mode:str = 'all' , start_date:str = None, end_date:str = None):
        """Initializes the Aggregation class.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to be aggregated.
            
        variable : str
            The variable/column in the DataFrame to be aggregated.
            
        time_step : str
            The time step for aggregation. Supported values: 'month', 'dekad', 'week'.
            
        metrics : List[str]
            The list of metrics to be applied during aggregation. Supported values: 'mean', 'median', 'min', 'max'.
        
        mode : str
            Perform aggregation on the entire dataset or on a subset of the dataset. Supported values: 'all', 'subset'.
        
        start_date : str
            The start date for the aggregation in case of subset mode. The format should be 'YYYY-MM-DD'.
        
        end_date : str
            The end date for the aggregation in case of subset mode. The format should be 'YYYY-MM-DD'.
            
        """
        self.original_df = df
        self.variable = variable
        self.time_step = time_step
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.valid_time_steps = ["month", "dekad", "week"]
        self._validate_input()
        # Resampling the data to daily frequency and removing NaN values
        self.df = pd.DataFrame(self.original_df[self.variable]).resample('D').mean().dropna()
        self.df = self._set_up_mode()
        # Creating an empty DataFrame to store the aggregated data
        self.resulted_df = pd.DataFrame()
        
    def _set_up_mode(self):
        """
        Filters the DataFrame based on specified time/date conditions.

        Returns:
        --------
        pd.DataFrame
            The filtered DataFrame.
        """
        if self.mode == 'all':
            return self.df
        elif self.mode == 'subset':
            try:
                if self.start_date and self.end_date:
                    return self.df.loc[self.start_date:self.end_date]
            except TypeError as e:
                print(e)
                raise TypeError("In case of subset mode: start and end dates should be provided in the format 'YYYY-MM-DD'.")
        
                
            
        
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
        
    def _validate_variable(self):
        """
        Validates the variable to be aggregated.
        
        Raises:
        -------
        ValueError:
            If the variable is not found in the input DataFrame columns.
            
        """
        if self.variable not in self.original_df.columns:
            raise ValueError(f"Variable '{self.variable}' not found in the input DataFrame columns.")
    
    def _validate_input(self):
        """
        Validates the input parameters.
        
        Raises:
        -------
        ValueError:
            If the time step or variable is invalid.

        """
        self._validate_time_step()
        self._validate_variable()

    def aggregate(self):
        """
        Aggregates the data based on the specified .
        """
        pass


class MonthlyAggregator(Aggregator):
    """
    Aggregates the time series data based on month-based time step.     
    """

    def __init__(self, df, variable):
        super().__init__(df, variable, 'month')

    def aggregate(self):
        self.resulted_df[f"{self.variable}-avg"] = self.df.groupby([self.df.index.year, self.df.index.month])[self.variable].transform('mean')
        return self.resulted_df

class DekadalAggregator(Aggregator):
    """
    Aggregates the data based on dekad-based time step.
    """

    def __init__(self, df, variable):
        super().__init__(df, variable, 'dekad')

    def aggregate(self):
        self.df['dekad'] = self.df.index.map(lambda x: 5 if x.day <= 10 else 15 if x.day <= 20 else 25)
        self.resulted_df['dekad'] = self.df['dekad']
        self.resulted_df[f"{self.variable}-avg"] = self.df.groupby([self.df.index.year, self.df.index.month, self.df['dekad']])[self.variable].transform('mean')
        return self.resulted_df

class WeeklyAggregator(Aggregator):
    """
    Aggregates the time series data based on week-based time step.
    """

    def __init__(self, df, variable):
        super().__init__(df, variable, 'week')

    def aggregate(self):
        self.resulted_df[f"{self.variable}-avg"] = self.df.groupby([self.df.index.year, self.df.index.isocalendar().week])[self.variable].transform('mean')
        return self.resulted_df
        


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

    time_step: str
        The time step for aggregation. Supported values: 'month', 'dekad', 'week'.

    metrics: List[str]
        The list of metrics to be applied during aggregation. Supported values: 'mean', 'median', 'min', 'max'.

    resulted_df: pd.DataFrame
        The resulting DataFrame storing the aggregated data.

    climatology_df: pd.DataFrame
        The DataFrame containing climatology information.

    Methods:
    -------
    aggregate:
        Aggregates the data based on the time step and metrics provided.

    filter_df:
        Filters the DataFrame based on specified time/date conditions.

    climatology:
        Calculates climatology based on the aggregated data.
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str] ,
                 mode:str = 'all' , start_date:str = None, end_date:str = None):
        """
        Initializes the Climatology class.
        """
        self.metrics = metrics
        self.valid_metrics = ["mean", "median", "min", "max"]
        super().__init__(df, variable, time_step , mode , start_date , end_date)
        self.climatology_df = pd.DataFrame()
        
        
    def _validate_metrics(self):
        """
        Validates the metrics to be used in the climatology computation.

        Raises:
            ValueError: If the metric is not one of the supported values.
            
        """
        for metric in self.metrics:
            if metric not in self.valid_metrics:
                raise ValueError(f"Invalid metric '{metric}'. Supported values: {self.valid_metrics}.")
            
    def _validate_input(self):
        super()._validate_input()
        self._validate_metrics()
          
    def __repr__(self):
        return f"Climatology(df={self.original_df}, variable={self.variable}, time_step={self.time_step}, metrics={self.metrics})"

    @staticmethod
    def _filter_df(df: pd.DataFrame = None, year: Union[int, None] = None,
                   month: Union[int, None] = None, dekad: Union[int, None] = None,
                   week: Union[int, None] = None, start_date: Union[str, None] = None,
                   end_date: Union[str, None] = None) -> pd.DataFrame:
        """
        Filters the DataFrame based on specified time/date conditions.

        Parameters:
        -----------
        df: pd.DataFrame, optional
            The DataFrame to be filtered.

        year: int or None, optional
            The year to filter the DataFrame.

        month: int or None, optional
            The month to filter the DataFrame.

        dekad: int or None, optional
            The dekad to filter the DataFrame.

        week: int or None, optional
            The week to filter the DataFrame.

        start_date: str or None, optional
            The start date for filtering.

        end_date: str or None, optional
            The end date for filtering.

        Returns:
        --------
        pd.DataFrame
            The filtered DataFrame.
            
        """
        if df is None:
            print("No dataframe provided")
            return pd.DataFrame()

        if start_date and end_date:
            df = df.loc[start_date:end_date]

        if year:
            df = df[df.index.year == year]

        if month:
            df = df[df.index.month == month]

        if dekad:
            df = df[df['dekad'] == dekad]

        if week:
            df = df[df.index.isocalendar().week == week]

        return df
    
    def aggregate(self):
        """
        Aggregates the data based on the time step and metrics provided.
        
        """
        if self.time_step == 'month':
             return MonthlyAggregator(self.df, self.variable).aggregate().drop_duplicates()
        elif self.time_step == 'dekad':
             return DekadalAggregator(self.df, self.variable).aggregate().drop_duplicates()
        elif self.time_step == 'week':
             return WeeklyAggregator(self.df, self.variable).aggregate().drop_duplicates()
        
    
    
    def _group_by(self , df):
        """
        Groups the DataFrame based on the provided time step.
        
        parameters:
        ----------
        
        df: pd.DataFrame
            The DataFrame to be grouped.
        
        returns:
        --------
        list
            The list of date parameters to be used for grouping.
        """
        
        if self.time_step == 'month':
            return [df.index.month]
        
        elif self.time_step == 'week':
            return [df.index.isocalendar().week]
        
        elif self.time_step == 'dekad':
            return [df['dekad'] , df.index.month]
        

    def compute_climatology(self, **kwargs) -> pd.DataFrame:
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
        self.climatology_df = self.aggregate()
        groupby_param = self._group_by(self.climatology_df)
        
        for metric in self.metrics:
            self.climatology_df[f'normal-{metric}'] = self.climatology_df.groupby(groupby_param)[f"{self.variable}-avg"].transform(metric)
            
        
        return self._filter_df(self.climatology_df, **kwargs)
    

if __name__ == "__main__":
    
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Morocco
    lat = 33.201
    lon = -7.373
        
    sm_ts = extract_obs_ts((lon, lat), ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    
    print(Climatology(sm_ts, "sm", "dekad", ["mean"]).compute_climatology(year = 2022))    


    