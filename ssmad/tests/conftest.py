"""
Module for defining fixtures and configurations used across tests.
"""

import pytest
import pandas as pd
import numpy as np
from ssmad.climatology import Climatology

@pytest.fixture
def data_sample():
    """
    Fixture providing a sample DataFrame for testing purposes.
    The sample data is read from a CSV file and formatted accordingly.
    """
    data_sample = pd.read_csv("ssmad/tests/data_sample.csv")
    # set the index to datetime index
    data_sample["Unnamed: 0"] = pd.to_datetime(data_sample["Unnamed: 0"])
    data_sample.set_index("Unnamed: 0", inplace=True)
    data_sample.index.name = None
    return data_sample 

@pytest.fixture
def climatology_sample(data_sample):
    """
    Fixture providing a sample climatology dataframe for testing purposes.
    """
    return Climatology(data_sample, time_step="month", variable='sm').compute_normals()

@pytest.fixture
def ascat_sm():
    """
    Fixture providing a sample ASCAT time series SSM for testing purposes.
    """    
    return [32.357273, 36.873528, 26.126818, 30.512056,34.7725]
              
    

if __name__ == '__main__':
    pytest.main([__file__])
