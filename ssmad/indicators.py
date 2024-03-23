

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm , beta , gamma


def zscore(obs, mean=None, std=None):
    """
    Computes the standardized z-score of the time series data.
    
    Parameters:
    ----------
    obs: pd.Series or np.ndarray or sequence-like object
        The time series data.
        
    mean: float, pd.Series or np.ndarray or sequence-like object, optional
        The mean of the distribution of the time series data. If None, it will be computed from `obs`.
    
    std:  float, pd.Series or np.ndarray or sequence-like object, optional
        The standard deviation of the distribution of the time series data. If None, it will be computed from `obs`.
        
    Returns:
    -------
    pd.Series or np.ndarray
        The z-score of the time series data.
    """
    # Convert to numpy array if not already a numpy array or a pandas Series
    obs = np.asarray(obs)
    mean = np.asarray(mean) if mean is not None else np.mean(obs)
    std = np.asarray(std) if std is not None else np.std(obs)
        
    return (obs - mean) / std

def smapi(obs, ref = None, metric = 'mean'):
    """
    Computes anomalies in time series data based on the Soil Moisture Anomaly Percent Index(SMAPI) method.
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    ref: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term mean (μ​) or median (η) of the variable(the climate normal)
    
    metric: str, optional
        The metric to be used for computing the anomalies. Supported values: 'mean', 'median'
    """

    obs = np.asarray(obs)
    ref = np.asarray(ref) if ref is not None else np.mean(obs) if metric == 'mean' else np.median(obs) 
    
    return ((obs - ref) / ref) * 100


def smd(obs , median = None, minimum = None, maximum = None):
    """
    Computes the Soil Moisture Deficit (SD) based on observed value and long-term median, minimum, and maximum values.
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    median: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term median of the variable. if None, it will be computed from `obs`.
        
    minimum: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term minimum of the variable. if None, it will be computed from `obs`.
        
    maximum: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term maximum of the variable. if None, it will be computed from `obs`.
        
    
    Returns:
    --------
    numpy.ndarray
        The Soil Moisture Deficit Index computed based on the given observed value(s).
    """
    obs = np.asarray(obs)
    median = np.asarray(median) if median is not None else np.median(obs)
    minimum = np.asarray(minimum) if minimum is not None else np.min(obs)
    maximum = np.asarray(maximum) if maximum is not None else np.max(obs)
    
    sd = np.where(obs > median,
                  (100 * ((obs - median) / (maximum - median))),
                  (100 * ((obs - median) / (median - minimum))))
    
    return sd

def smdi(sd):
    """
    Computes the Soil Moisture Deficit Index (SMDI) incrementally based on the Soil Moisture Deficit (SD) values.
    """
    sd = np.asarray(sd)
    sd = sd /50
    for i in range(1, len(sd)):
        sd[i] += 0.5 * sd[i-1]  
        
    return sd



def smad(obs, median = None , iqr= None):
    """
    Computes the anomalies in time series data based on the Standardized Median Absolute Deviation(SMAD) method.
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    median: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term median of the variable. if None, it will be computed from `obs`.
        
    iqr: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term interquartile range of the variable. if None, it will be computed from `obs`.
        
        
    Returns:
    --------
    numpy.ndarray
        The anomalies computed based on the given observed value(s) and the long-term median.
    """
    obs = np.asarray(obs)
    median = np.asarray(median) if median is not None else np.median(obs)
    iqr = np.asarray(iqr) if iqr is not None else np.percentile(obs, 75) - np.percentile(obs, 25)
    
    return (obs - median) / iqr

def smca(obs, metric= 'mean' , ref= None , minimum= None, maximum= None):
    """
    Computes the anomalies in time series data based on the Soil Moisture Content Anomaly(SMCA) method.
        
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    metric: str, optional
        The metric to be used for computing the anomalies. Supported values: 'mean', 'median'
        
    ref: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term mean (μ​) or median (η) of the variable(the climate normal)
        
    minimum: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term minimum of the variable. if None, it will be computed from `obs`.
        
    maximum: float, pd.Series or np.ndarray or sequence-like object, optional
        The long-term maximum of the variable. if None, it will be computed from `obs`.
    

    Returns:
    --------
    numpy.ndarray
        The anomalies computed based on the given observed value(s) and the long-term median.
    """
    obs = np.asarray(obs)
    ref = np.asarray(ref) if ref is not None else np.mean(obs) if metric == 'mean' else np.median(obs)
    minimum = np.asarray(minimum) if minimum is not None else np.min(obs)
    maximum = np.asarray(maximum) if maximum is not None else np.max(obs)
    
    return ((obs - ref) / (maximum - minimum)) 


def smci(obs, minimum = None, maximum = None):
    """
    Computes the anomalies in time series data based on the Soil Moisture Condition Index(SMCI) method.
        
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    minimum: float, pd.Series or np.ndarray or sequence-like object
        The long-term minimum of the variable. if None, it will be computed from `obs`.
        
    maximum: float, pd.Series or np.ndarray or sequence-like object
        The long-term maximum of the variable. if None, it will be computed from `obs`.
        

    Returns:
    --------
    numpy.ndarray
        The Soil Moisture Content Index computed based on the given observed value(s).
    """
    obs = np.asarray(obs)
    minimum = np.asarray(minimum) if minimum is not None else np.min(obs)
    maximum = np.asarray(maximum) if maximum is not None else np.max(obs)
    
    return ((obs - minimum) / (maximum - minimum)) 


def smds(obs):
    """
    Computes anomalies in time series data based on the Soil Moisture Drought Severity(SMDS) method.
    SMDS = 1 - SMP
    SMP = (rank(x) / (n+1))
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    Returns:
    --------
    numpy.ndarray
        The Soil Moisture Drought Severity computed based on the given observed value(s).
    """
    obs = np.asarray(obs)
    smp = (np.argsort(np.argsort(obs)) + 1 )/ (len(obs) + 1)
    
    return 1- smp

def essmi(obs):
    """
    Compute the anomalies in time series data based on the Empirical Standardized Soil Moisture Index(ESSMI) method.
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    returns:
    --------
    
    numpy.ndarray
        The Empirical Standardized Soil Moisture Index computed based on the given observed value(s).
    """
    
    obs = np.asarray(obs)
    kde = gaussian_kde(obs, bw_method= "scott")
    ecdf = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in obs])
    
    return norm.ppf(ecdf)


def para_dis(obs, dist = 'beta'):
    
    """
    Compute the anomalies in time series data based on fitting the observed data to a parametric distribution.
    
    parameters:
    ----------
    
    obs: pd.Series or np.ndarray or sequence-like object
        The observed time series data.
        
    dist: str, optional
        The distribution to fit the observed data to. Supported values: 'beta', 'gamma'
    """
    
    obs = np.asarray(obs)
    
    if dist  == 'beta':
        a,b,loc,scale = beta.fit(obs)
        fitted = beta(a,b,loc,scale)
        
    elif dist == 'gamma':
        shape,loc,scale = gamma.fit(obs)
        fitted = gamma(shape,loc,scale)
        
    cdf = fitted.cdf(obs)
    
    return norm.ppf(cdf)



if __name__ == '__main__':
    
    sm = np.random.rand(10)
    sm = np.array(sm)
    mean = np.mean(sm)
    median = np.median(sm)
    mini = np.min(sm)
    maxi = np.max(sm)
    
    print(para_dis(sm, 'beta'))
    
    
