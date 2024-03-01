"""
A module for calculating climatology based on aggregation.
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
    """
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str]):
        """
        Initializes the Aggregation class.

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
        self.df = df
        self.variable = variable
        self.time_step = time_step
        self.metrics = metrics
        self.resulted_df = pd.DataFrame()

    def aggregate(self):
        """
        Aggregates the data based on the specified time step and metrics.
        """
        pass


class MonthlyAggregation(Aggregation):
    """
    Aggregates the data based on month-based time step.
    """

    def __init__(self, df, variable, time_step, metrics):
        super().__init__(df, variable, time_step, metrics)

    def aggregate(self):
        for metric in self.metrics:
            self.resulted_df[metric] = self.df.groupby([self.df.index.year, self.df.index.month]).transform(metric)
        return self.resulted_df

class DekadalAggregation(Aggregation):
    """
    Aggregates the data based on dekad-based time step.
    """

    def __init__(self, df, variable, time_step, metrics):
        super().__init__(df, variable, time_step, metrics)

    def aggregate(self):
        self.df['dekad'] = self.df.index.map(lambda x: 5 if x.day <= 10 else 15 if x.day <= 20 else 25)
        for metric in self.metrics:
            self.df[metric] = self.df.groupby([self.df.index.year, self.df.index.month, self.df['dekad']])[self.variable].transform(metric)
        self.df = self.df.drop(columns=self.variable)
        return self.df

class WeeklyAggregation(Aggregation):
    """
    Aggregates the data based on week-based time step.
    """

    def __init__(self, df, variable, time_step, metrics):
        super().__init__(df, variable, time_step, metrics)

    def aggregate(self):
        for metric in self.metrics:
            self.resulted_df[metric] = self.df.groupby([self.df.index.year, self.df.index.isocalendar().week]).transform(metric)
        return self.resulted_df

class Aggregator(Aggregation):
    """
    Aggregates the data based on the specified time step and metrics.

    Methods:
    --------
    aggregate():
        Aggregates the data based on the specified time step and metrics and removes duplicates.
    """
    def __init__(self, df, variable, time_step, metrics):
        super().__init__(df, variable, time_step, metrics)

    def aggregate(self):
        if self.time_step == 'month':
            return MonthlyAggregation(self.df, self.variable, self.time_step, self.metrics).aggregate().drop_duplicates()
        elif self.time_step == 'dekad':
            return DekadalAggregation(self.df, self.variable, self.time_step, self.metrics).aggregate().drop_duplicates()
        elif self.time_step == 'week':
            return WeeklyAggregation(self.df, self.variable, self.time_step, self.metrics).aggregate().drop_duplicates()
        else:
            raise ValueError("Invalid time step. Please choose from 'month', 'dekad', 'week'")




class Climatology(Aggregation):
    """
    A class for calculating climatology based on aggregation.

    Attributes:
    ----------
    df_original: pd.DataFrame
        The original input DataFrame.

    df: pd.DataFrame
        The input DataFrame containing the data to be aggregated.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    time_step: str
        The time step for aggregation ('month', 'dekad', or 'week').

    metrics: List[str]
        The list of metrics to be applied during aggregation. Supported values: 'mean', 'median', 'min', 'max'.

    resulted_df: pd.DataFrame
        The resulting DataFrame after aggregation.

    climatology_df: pd.DataFrame
        The DataFrame containing climatology information.

    Methods:
    -------
    aggregate:
        Aggregates the data based on the time step and metrics provided.

    filter_df:
        Filters the DataFrame based on specified conditions.

    climatology:
        Calculates climatology based on the aggregated data.
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str]):
        """
        Initializes the Climatology class.
        """
        super().__init__(df, variable, time_step, metrics)
        self.df_original = df
        self.df = pd.DataFrame(df[variable]).resample("1D").mean().dropna()
        self.climatology_df = pd.DataFrame()

    @staticmethod
    def _filter_df(df: pd.DataFrame = None, year: Union[int, None] = None,
                   month: Union[int, None] = None, dekad: Union[int, None] = None,
                   week: Union[int, None] = None, start_date: Union[str, None] = None,
                   end_date: Union[str, None] = None) -> pd.DataFrame:
        """
        Filters the DataFrame based on specified conditions.

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
        return Aggregator(self.df, self.variable, self.time_step, self.metrics).aggregate()
    
    def climatology(self, **kwargs) -> pd.DataFrame:
        """
        Calculates climatology based on the aggregated data.

        Parameters:
        -----------
        kwargs:
            Additional time filtering parameters.

        Returns:
        --------
        pd.DataFrame
            The DataFrame containing climatology information.
        """
        self.climatology_df = self.aggregate()
        
        for metric in self.metrics:
            if self.time_step == 'month':
                self.climatology_df[f'normal_{metric}'] = self.climatology_df.groupby([self.climatology_df.index.month])[metric].transform(metric)
            elif self.time_step == 'week':
                self.climatology_df[f'normal_{metric}'] = self.climatology_df.groupby([self.climatology_df.index.isocalendar().week])[metric].transform(metric)
            elif self.time_step == 'dekad':
                self.climatology_df[f'normal_{metric}'] = self.climatology_df.groupby([self.climatology_df.index.month, self.climatology_df['dekad']])[metric].transform(metric)
        
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

    
    climatology = Climatology(sm_ts, "sm", "month", ["mean" , "median"])
    print(climatology.climatology(month = 2))