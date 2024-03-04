"""
A module for calculating climatology for different time steps (month, dekad, week) 
using different metrics (mean, median, min, max).
"""

__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

import pandas as pd
from typing import List, Union

class Aggregation:
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
    """
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str):
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
            
        """
        self.original_df = df
        self.variable = variable
        self.time_step = time_step
        self.valid_time_steps = ["month", "dekad", "week"]
        self._validate_input()
        # Resampling the data to daily frequency and removing NaN values
        self.df = pd.DataFrame(self.original_df[self.variable]).resample('D').mean().dropna()
        # Creating an empty DataFrame to store the aggregated data
        self.resulted_df = pd.DataFrame()
        
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


class MonthlyAggregation(Aggregation):
    """
    Aggregates the time series data based on month-based time step.     
    """

    def __init__(self, df, variable):
        super().__init__(df, variable, 'month')

    def aggregate(self):
        self.resulted_df[f"{self.variable}-avg"] = self.df.groupby([self.df.index.year, self.df.index.month])[self.variable].transform('mean')
        return self.resulted_df

class DekadalAggregation(Aggregation):
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

class WeeklyAggregation(Aggregation):
    """
    Aggregates the time series data based on week-based time step.
    """

    def __init__(self, df, variable):
        super().__init__(df, variable, 'week')

    def aggregate(self):
        self.resulted_df[f"{self.variable}-avg"] = self.df.groupby([self.df.index.year, self.df.index.isocalendar().week])[self.variable].transform('mean')
        return self.resulted_df

class Aggregator(Aggregation):
    """
    Aggregates the time series data based on the specified time step.

    Methods:
    --------
    aggregate():
        Aggregates the data based on the specified time step and removes duplicates.
    """
    def __init__(self, df, variable, time_step):
        super().__init__(df, variable, time_step)
        

    def aggregate(self ):
        if self.time_step == 'month':
            return MonthlyAggregation(self.df, self.variable).aggregate().drop_duplicates()
        elif self.time_step == 'dekad':
            return DekadalAggregation(self.df, self.variable).aggregate().drop_duplicates()
        elif self.time_step == 'week':
            return WeeklyAggregation(self.df, self.variable).aggregate().drop_duplicates()
        


class Climatology(Aggregation):
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
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str]):
        """
        Initializes the Climatology class.
        """
        self.metrics = metrics
        self.valid_metrics = ["mean", "median", "min", "max"]
        super().__init__(df, variable, time_step )
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
        return Aggregator(self.df, self.variable, self.time_step).aggregate()
    
    
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
        

    def climatology(self, **kwargs) -> pd.DataFrame:
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
    
    # Example usage
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Morocco
    lat = 33.201
    lon = -7.373
    
    sm_ts = extract_obs_ts((lon, lat), ascat_path, obs_type="sm" , read_bulk=False)["ts"]

    
    climatology = Climatology(sm_ts, "sm", "dekad", ["mean",'median','min','max'])
    print(climatology.climatology(month = 2))