"""
A module for soil moisture anomalies calculation methods based on climatology. The module contains the following classes:

1. AnomalyDetector: A base class for detecting anomalies in time series data based on climatology. Inherited from the Climatology class.
2. Z_Score: A class for detecting anomalies in time series data based on the Z-Score method.
3. SMAPI: A class for detecting anomalies in time series data based on the Soil Moisture Anomaly Percent Index(SMAPI) method.
4. SMDI: A class for detecting anomalies in time series data based on the Soil Moisture Deficit Index(SMDI) method.
5. ESSMI: A class for detecting anomalies in time series data based on the Empirical Standardized Soil Moisture Index(ESSMI) method.
6. SMAD: A class for detecting anomalies in time series data based on the Standardized Median Absolute Deviation(SMAD) method.

"""

__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"


from ssmad.climatology import *
import pandas as pd
import numpy as np
from typing import List
from scipy.stats import gaussian_kde
from scipy.stats import norm


class AnomalyDetector(Climatology):
    """
    A base class for detecting anomalies in time series data based on climatology. Inherited from the Climatology class.
    
    parameters:
    -----------
    df: pd.DataFrame
        A dataframe containing the time series data.
        
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
        
    metrics: List[str]
        A list of metrics to be used in the climate normal(climatology) computation. It can be any of the following:
        ['mean', 'median', 'min', 'max']
    """
    

    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str]):
        """
        Initialize the AnomalyDetector class.
        
        parameters:
        -----------
        df: pd.DataFrame
            A dataframe containing the time series data.
        
        variable: str
            The name of the variable in the time series data to be analyzed.
        
        time_step: str
            The time step of the time series data. It can be any of the following:
            [ 'week','dekad','month']
        
        metrics: List[str]
            A list of metrics to be used in the climate normal(climatology) computation. It can be any of the following:
            ['mean', 'median', 'min', 'max']
        
        anomaly_df: pd.DataFrame
            A dataframe containing the computed anomalies.
            
        groupby_param: List[str]
            The column name to be used for grouping the data for the computation of the climate normal.
            
        """
        super().__init__(df, variable, time_step, metrics)
        self.anomaly_df = pd.DataFrame()
        self.groupby_param = None  
                
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data before computing the anomalies.
        
        parameters:
        -----------
        kwargs: str
            Date/time parameters to be used for filtering the data before computing the anomalies. It can be any of the following:
            ['year', 'month', 'week', 'dekad' , start_date, end_date]
            
        returns:
        --------
        
        pd.DataFrame
            A dataframe containing the preprocessed data for computing the anomalies.
        
        """
        self._validate_input()
        self.climatology_df = self.climatology(**kwargs)
        self.groupby_param = self._group_by(self.climatology_df)
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        """
        Detect the anomalies in the time series data.
        
        parameters:
        -----------
        kwargs: str
            Date/time parameters to be used for filtering the data before computing the anomalies. It can be any of the following:
            ['year', 'month', 'week', 'dekad' , start_date, end_date]
            
        returns:
        --------
        
        pd.DataFrame
            A dataframe containing the computed anomalies.
        
        """
        self.anomaly_df = self._filter_df(self._preprocess(), **kwargs).copy()
        
        
class Z_Score(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Z-Score method. 
    
    z_score = (x - μ) / σ
        
        where:
        x: the average value of the variable in the time series data. It can be any of the following:
        Daily average, weekly average, monthly average, etc.
        μ: the long-term mean of the variable(the climate normal).
        σ: the long-term standard deviation of the variable.
        
    parameters:
    -----------
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
    
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str ):
        super().__init__(df, variable, time_step, ["mean"])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        self.climatology_df['normal-std'] = self.climatology_df.groupby(self.groupby_param)[f"{self.variable}-avg"].transform('std')
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.anomaly_df['z_score'] = (self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df['normal-mean']) / self.anomaly_df['normal-std']
        return self.anomaly_df
    


class SMAPI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Anomaly Percent Index(SMAPI) method.
    
    SMAPI = ((x - ref) / ref) * 100
    
    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    ref: the long-term mean (μ​) or median (η) of the variable(the climate normal). 
    
    parameters:
    -----------
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
        
    metrics: List[str]
        A list of metrics to be used in the climate normal(climatology) computation. It can be any of the following:
        ['mean', 'median']
        
    """
    
    def __init__(self, df: pd.DataFrame, variable: str , time_step: str, metrics: List[str] = ["mean",'median']):
        super().__init__(df, variable, time_step, metrics)
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        for metric in self.metrics:
            self.anomaly_df[f'smapi-{metric}'] = ((self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df[f'normal-{metric}']) / self.anomaly_df[f'normal-{metric}']) * 100
        return self.anomaly_df


class SMDI(AnomalyDetector):
    """ 
    A class for detecting anomalies in time series data based on the Soil Moisture Deficit Index(SMDI) method.
    
    SMDI = 0.5 * SMDI(t-1) + (SD(t) / 50)
    
    where 
    
    SD(t) = ((x - η) / (η - min)) * 100 if x <= η
    SD(t) = ((x - η) / (max - η)) * 100 if x > η
    
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    η: the long-term median of the variable(the climate normal).
    min: the long-term minimum of the variable.
    max: the long-term maximum of the variable.
    t: the time step of the time series data.
    
    
   
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str = 'week'):
        super().__init__(df, variable, time_step, ['mean', 'median', 'min', 'max'])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        self.climatology_df['SD-condition'] = self.climatology_df[f"{self.variable}-avg"] <= self.climatology_df['normal-median']
        return self.climatology_df
    
    def _compute_soil_deficit(self, df: pd.DataFrame ) -> pd.DataFrame:
        """Compute the Soil Moisture Deficit(SD) for the time series data based on the climate normal values [median, min, max].
        
        """
        df['SD'] = np.where(
            df[f"{self.variable}-avg"] <= df['normal-median'],
            ((df[f"{self.variable}-avg"] - df['normal-median']) / (df['normal-median'] - df['normal-min'])) * 100,
            ((df[f"{self.variable}-avg"] - df['normal-median']) / (df['normal-max'] - df['normal-median'])) * 100)
        
        return df
    
    def _compute_soil_deifict_index(self, df: pd.DataFrame ) -> pd.DataFrame:
        """
        Compute the Soil Moisture Deficit Index(SMDI) for the time series data based on the Soil Moisture Deficit(SD).
        
        """
        
        df['SMDI'] = 0.0
        df.reset_index(inplace=True)
        df['SMDI'] = 0.5 * df['SMDI'].shift(1) + df['SD'] / 50
        df.loc[0, 'SMDI'] = df.loc[0, 'SD'] / 50
        for i in range(1, len(df)):
            df.loc[i, 'SMDI'] = 0.5 * df.loc[i-1, 'SMDI'] + df.loc[i, 'SD'] / 50
        return df
    
    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the computed anomalies by removing the unnecessary columns and setting the index.
        """
        df.set_index('index', inplace=True)
        df.drop(columns=['SD-condition'], inplace=True)
        df.index.name = None
        return df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.anomaly_df = self._compute_soil_deficit(self.anomaly_df)
        self.anomaly_df = self._compute_soil_deifict_index(self.anomaly_df)
        self.anomaly_df = self._finalize(self.anomaly_df)

        return self.anomaly_df
    
 
    

class ESSMI(AnomalyDetector):
    """ 
    A class for detecting anomalies in time series data based on the Empirical Standardized Soil 
    Moisture Index(ESSMI) method.
        
    The index is computed by fitting the nonparametric empirical probability
    density function (ePDF) using the kernel density estimator KDE
    
    f^h = 1/nh * Σ K((x - xi) / h)
    K = 1/√(2π) * exp(-x^2/2)
    
    where:
    f^h: the ePDF
    K: the Guassian kernel function
    h: the bandwidth of the kernel function as smoothing parameter (Scott's rule)
    n: the number of observations
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    xi: the ith observation
    
    The ESSMI is then computed by transforming the ePDF to the standard normal distribution with a mean of zero and
    a standard deviation of one using the inverse of the standard normal distribution function.      
    
    ESSMI = Φ^-1(F^h(x))
        
        where:
        Φ^-1: the inverse of the standard normal distribution function
        F^h: the ePDF
        
    parameters:
    -----------
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']

    
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str):
        super().__init__(df, variable, time_step, ['mean'])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        self.climatology_df['ecdf'] = np.nan
        self.climatology_df['z'] = np.nan
        
        data_groups = self.climatology_df.groupby(self.groupby_param)[f"{self.variable}-avg"]
        for _, group in data_groups:
            temp_df = self._compute_ecdf(group)
            self.climatology_df.update(temp_df)        
        return self.climatology_df
    
    def _compute_ecdf(self, group: pd.Series ) -> pd.Series:
        """
        Compute the empirical cumulative distribution function (eCDF) for the time series data.
        """
        
        temp_df = pd.DataFrame(group)
        temp_df['ecdf'] = np.nan
        temp_df['z'] = np.nan
        kde = gaussian_kde(temp_df[f"{self.variable}-avg"], bw_method="scott")
        temp_df['ecdf'] = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in temp_df[f"{self.variable}-avg"]])
        return temp_df

    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.anomaly_df['z'] = norm.ppf(self.anomaly_df['ecdf'])    
        return self.anomaly_df
    
class SMAD(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Standardized Median Absolute Deviation(SMAD) method.
    
    SMAD = (x - η) / IQR
    
    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    η: the long-term median of the variable(the climate normal).
    IQR: the interquartile range of the variable. It is the difference between the 75th and 25th percentiles of the variable.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str):
        super().__init__(df, variable, time_step, ['mean','median'])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        self.groupby_param = self._group_by(self.climatology_df)
        self.climatology_df['IQR'] = self.climatology_df.groupby(self.groupby_param)[f"{self.variable}-avg"].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))        
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)   
        self.anomaly_df['SMAD'] = (self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df['normal-median']) / self.anomaly_df['IQR']     
        return self.anomaly_df
    
class SMDS(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Drought Severity(SMDS) method.
    
    SMDS = 1 - SMP
    SMP = (rank(x) / (n+1))
    
    where:
    
    SMP: the Soil Moisture Percentile. It is the percentile of the average value of the variable in the time series data.
    SMDS: the Soil Moisture Drought Severity. It is the severity of the drought based on the percentile of the average value of the variable in the time series data.
    rank(x): the rank of the average value of the variable in the time series data.
    n: the number of years in the time series data.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        A dataframe containing the time series data.
        
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
    
    metrics: List[str]
        A list of metrics to be used in the climate normal(climatology) computation. It can be any of the following:
        ['mean', 'median']
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str, metrics: List[str]):
        super().__init__(df, variable, time_step, metrics)
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)  
        for metric in self.metrics:
            self.climatology_df[f"rank-{metric}"] = self.climatology_df.groupby(self.groupby_param)[f"{self.variable}-avg"].rank()   
            self.climatology_df[f"SMP-{metric}"] = (self.climatology_df[f"rank-{metric}"] / (len(self.climatology_df.groupby(self.climatology_df.index.year)) +1))
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        for metric in self.metrics:
             self.anomaly_df[f'SMDS-{metric}'] = 1 - self.anomaly_df[f"SMP-{metric}"] 
        return self.anomaly_df
    
class SMCI(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Condition Index(SMCI) method.
    
    SMCI = ((x - min) / (max - min)) * 100
    
    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    min: the long-term minimum of the variable.
    max: the long-term maximum of the variable.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
    """
    
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str):
        super().__init__(df, variable, time_step, ['mean' , 'median' , 'min' , 'max'])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.anomaly_df['SMCI'] = ((self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df['normal-min']) / (self.anomaly_df['normal-max'] - self.anomaly_df['normal-min'])) * 100
        return self.anomaly_df
    
class SMCA(AnomalyDetector):
    """
    A class for detecting anomalies in time series data based on the Soil Moisture Content Anomaly(SMCA) method.
    
    SMCA = (x - μ) / (max - μ) if usingt the long-term mean(μ) as the climate normal
    SMCA = (x - η) / (max - η) if using the long-term median(η) as the climate normal
    
    where:
    x: the average value of the variable in the time series data. It can be any of the following:
    Daily average, weekly average, monthly average, etc.
    
    μ: the long-term mean of the variable(the climate normal).
    η: the long-term median of the variable(the climate normal).
    max: the long-term maximum of the variable.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        A dataframe containing the time series data.
    
    variable: str
        The name of the variable in the time series data to be analyzed.
        
    time_step: str
        The time step of the time series data. It can be any of the following:
        [ 'week','dekad','month']
        
    """
        
    def __init__(self, df: pd.DataFrame, variable: str, time_step: str):
        super().__init__(df, variable, time_step, ['mean' , 'median' , 'min' , 'max'])
        
    def _preprocess(self, **kwargs) -> pd.DataFrame:
        super()._preprocess(**kwargs)
        return self.climatology_df
    
    def detect_anomaly(self, **kwargs) -> pd.DataFrame:
        super().detect_anomaly(**kwargs)
        self.anomaly_df['SMA-mean'] = ((self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df['normal-mean']) / (self.anomaly_df['normal-max'] - self.anomaly_df['normal-min'])) * 100
        self.anomaly_df['SMA-median'] = ((self.anomaly_df[f"{self.variable}-avg"] - self.anomaly_df['normal-median']) / (self.anomaly_df['normal-max'] - self.anomaly_df['normal-min'])) * 100
        return self.anomaly_df
    

    
if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from ssmad.data_reader import extract_obs_ts
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Morocco
    lat = 33.201
    lon = -7.373
    
    sm_ts = extract_obs_ts((lon, lat), ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    print(SMAD(sm_ts, "sm" , 'week').detect_anomaly(month = 5))

