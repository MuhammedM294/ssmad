"""
Module for defining fixtures and configurations used across tests.
"""

import pytest
import pandas as pd
from ssmad import climatology

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


