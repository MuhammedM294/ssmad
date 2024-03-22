from typing import Union , List
import pandas as pd



def fillna(df: pd.DataFrame, variable: str, fillna_window_size: int) -> pd.DataFrame:
    """
    Fills NaN values in the time series data using a moving window average.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be filled indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be filled.

    fillna_window_size: int
        The size of the moving window [days] for filling NaN values. It is recommended to be an odd number.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the filled time series data.
    """
    df[variable] = df[variable].fillna(df[variable].rolling(window=fillna_window_size , center=True , min_periods=1).mean())
    return df

def smooth(df: pd.DataFrame, variable: str, window_size: int) -> pd.DataFrame:
    """
    Smooths the time series data using a moving window average.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be smoothed indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be smoothed.

    window_size: int
        The size of the moving window [days] for smoothing(. It is recommended to be an odd number.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the smoothed time series data.
    """

    df[variable] = df[variable].rolling(window=window_size, center=True, min_periods=1).mean()
    return df


def filter_df(df: pd.DataFrame = None, year: Union[int, None] = None,
                   month: Union[int, None] = None, dekad: Union[int, None] = None,
                   bimonth: Union[int, None] = None,day: Union[int, None] = None,
                   week: Union[int, None] = None, start_date: Union[str, None] = None,
                   end_date: Union[str, None] = None) -> pd.DataFrame:
        """
        Filters the DataFrame based on specified time/date conditions.

        Parameters:
        -----------
        df: pd.DataFrame, optional
            The DataFrame to be filtered. It should be indexed by a datetime index.

        year: int or None, optional
            The year to filter the DataFrame.

        month: int or None, optional
            The month to filter the DataFrame.
        
        bimonth: int or None, optional
            The bimonth to filter the DataFrame.
            

        dekad: int or None, optional
            The dekad to filter the DataFrame.

        week: int or None, optional
            The week to filter the DataFrame.
            
        day: int or None, optional
            The day to filter the DataFrame.

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
            df = df.truncate(before=start_date, after=end_date)

        if year:
            df = df[df.index.year == year]

        if month:
            df = df[df.index.month == month]
        
        if bimonth:
            if bimonth not in [1, 2]:
                raise ValueError("Invalid bimonth value. Supported values: 1, 2")
            if 'bimonth' not in df.columns:
                df['bimonth'] = df.index.map(lambda x: 1 if x.day <= 15 else 2)
            df = df[df['bimonth'] == bimonth]

        if dekad:
            if dekad not in [1, 2, 3]:
                raise ValueError("Invalid dekad value. Supported values: 1, 2, 3")
            if 'dekad' not in df.columns:
                df['dekad'] = df.index.map(lambda x: 1 if x.day <= 10 else 2 if x.day <= 20 else 3)
            df = df[df['dekad'] == dekad]

        if week:
            df = df[df.index.isocalendar().week == week]
        
        if day:
            df = df[df.index.day == day]

        return df


def monthly_agg(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Aggregates the time series data based on month-based time step.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be aggregated indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the aggregated data.
    """
    return df.groupby([df.index.year, df.index.month])[variable].transform('mean').drop_duplicates()


def dekadal_agg(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Aggregates the time series data based on dekad-based time step.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be aggregated indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the aggregated data.
    """
    df['dekad'] = df.index.map(lambda x: 1 if x.day <= 10 else 2 if x.day <= 20 else 3)
    return df.groupby([df.index.year, df.index.month, 'dekad'])[variable].transform('mean').drop_duplicates()

def weekly_agg(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Aggregates the time series data based on week-based time step.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be aggregated indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the aggregated data.
    """
    return df.groupby([df.index.year, df.index.isocalendar().week])[variable].transform('mean').drop_duplicates()


def bimonthly_agg(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Aggregates the time series data based on bimonth-based time step.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be aggregated indexed by datetime index.

    variable: str
        The variable/column in the DataFrame to be aggregated.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the aggregated data.
    """
    if 'bimonth' not in df.columns:
        df['bimonth'] = df.index.map(lambda x: 1 if x.day <= 15 else 2)
    return df.groupby([df.index.year, df.index.month, df['bimonth']])[variable].transform('mean').drop_duplicates()


def clim_groupping(df: pd.DataFrame, time_step: str) -> list:
    """
        Groups the DataFrame based on the provided time step for climatology computation.
        
        parameters:
        ----------
        
        df: pd.DataFrame
            The DataFrame to be grouped.
        
        returns:
        --------
        list
            The list of date parameters to be used for grouping.
    """
        
    if time_step == 'month':
        return [df.index.month]
    
    elif time_step == 'week':
        return [df.index.isocalendar().week]
    
    elif time_step == 'dekad':
        
        if 'dekad' not in df.columns:
            df['dekad'] = df.index.map(lambda x: 1 if x.day <= 10 else 2 if x.day <= 20 else 3)
        return [df['dekad'] , df.index.month]
    
    elif time_step == 'bimonth':
        
        if 'bimonth' not in df.columns:
            df['bimonth'] = df.index.map(lambda x: 1 if x.day <= 15 else 2)
            
        return [df['bimonth'] , df.index.month]
    
    elif time_step == 'day':
        return [df.index.day, df.index.month]
    

def compute_clim(df: pd.DataFrame, time_step: str , variable: str , metrics: List[str]) -> pd.DataFrame:
    """
    Computes the climatology of the time series data based on the provided time step.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the time series data to be aggregated indexed by datetime index.

    time_step: str
        The time step to be used for computing the climatology. Supported values: 'month', 'week', 'dekad', 'bimonth', 'day'
        
    variable: str
        The variable/column in the DataFrame to be aggregated.
        
    metrics: List[str]
        The metrics to be computed. Supported values: 'mean', 'median', 'min', 'max',  etc.

    Returns:
    --------
    pd.DataFrame
        The DataFrame containing the climatology data.

    """

    for metric in metrics:
        if metric == 'iqr':
            df['norm-' + metric] = df.groupby(clim_groupping(df, time_step))[variable].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
        else:
            df['norm-' + metric] = df.groupby(clim_groupping(df, time_step))[variable].transform(metric)
        
    return df

if __name__ == "__main__":
    
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Morocco
    lat = 33.201
    lon = -7.373
        
    sm_ts = extract_obs_ts((lon, lat), ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    df = pd.DataFrame(sm_ts).resample("D").mean()
    
    monthly_df = monthly_agg(df, "sm")
    monthly_df = pd.DataFrame(monthly_df)
    monthly_df = monthly_df.rename(columns={"sm": "sm-avg"})
    
    x = compute_clim(monthly_df, "month", "sm-avg", ["mean","median","min","max"])
    
    print(filter_df(x, month= 12))
