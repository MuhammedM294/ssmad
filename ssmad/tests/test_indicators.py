import pytest
import pandas as pd
import numpy as np
from ssmad.indicators import (
    zscore,
    smapi,
    smdi,
    smad,
    smca,
    smci,
    smds,
    essmi,
    smd)



def test_zscore(ascat_sm):
    
    expected = [0.0619,  1.2848, -1.6251, -0.4376,  0.7159]
    resulted = zscore(ascat_sm)
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    

def test_smapi(ascat_sm):
    

    expected = [ 0.7122,  14.769,  -18.6800,  -5.0309,   8.2296]
    resulted = smapi(ascat_sm , metric='mean')
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    
    expected = [ 0, 13.9574, -19.2551,  -5.7026,  7.4642]
    resulted = smapi(ascat_sm , metric='median')
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    
    
def test_smdi(ascat_sm):
    
    expected = [0.0, 2.0, -1.0, -1.0923, 0.5234]
    sd = smd(ascat_sm)
    resulted = smdi(sd)
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    
    
def test_smad(ascat_sm):
    
    expected= [ 0, 1.0600,  -1.4623,   -0.4331,   0.5668]
    resulted = smad(ascat_sm)
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    

def test_smca(ascat_sm):
    
    expected = [ 0.0212,   0.4415,  -0.5584,  -0.1504,   0.2460]
    resulted = smca(ascat_sm, metric='mean')
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    
    expected = [ 0, 0.4202,  -0.5797,  -0.1717,   0.2247]
    resulted = smca(ascat_sm, metric='median')
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    

def test_smci(ascat_sm):
    
    expected = [0.5797,  1,  0, 0.4080,    0.8044 ]
    resulted = smci(ascat_sm)
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    
    
def test_smds(ascat_sm):
    
    expected = [0.5, 0.1666,  0.8333,  0.6666,  0.3333 ]
    resulted = smds(ascat_sm)
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)    
    
    
def test_essmi(ascat_sm):
    
    expected = [-0.0060,  0.9750, -1.1828, -0.3630,  0.4971]
    resulted = essmi(ascat_sm)
    
    np.testing.assert_allclose(resulted, expected, rtol=1e-4, atol= 1e-4 , verbose=True)
    

    
if __name__ == '__main__':
    pytest.main([__file__])