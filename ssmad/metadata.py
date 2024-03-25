import numpy as np
from ssmad.anomaly_detectors import (
    ZScore,
    SMAPI,
    SMDI,
    SMCA,
    SMAD,
    SMCI,
    SMDS,
    ESSMI,
    ParaDis)

# Supported anomaly detection methods
_Detectors = \
    {
    'zscore': ZScore,
    'smapi-mean': SMAPI,
    'smapi-median': SMAPI,
    'smdi': SMDI,
    'smca-mean': SMCA,
    'smca-median': SMCA,
    'smad': SMAD,
    'smci': SMCI, 
    'smds': SMDS,
    'essmi': ESSMI,
    'beta':ParaDis, 
    'gamma':ParaDis,
    }
    
indicators_thresholds= \
    {
    'zscore': {
               "D-3":(-3, -2),
               "D-2":(-2, -1.5),
               "D-1":(-1.5, -1),
               "NN":(-1, 1),
               "W-1":(1, 1.5),
               "W-2":(1.5, 2),
               "W-3":(2, 3)
               
               },
    
    'smapi': {
                "D-4":(-100, -50),
                "D-3":(-50, -30),
                "D-2":(-30, -15),
                "D-1":(-15, -5),
                "NN":(-5, 5),
                "W-1":(5, 15),
                "W-2":(15, 30),
                "W-3":(30, 50),
                "W-4":(50, 100)
                
                },
    
    'smdi': {
                "D-4":(-4, -3),
                "D-3":(-3, -2),
                "D-2":(-2, -1),
                "D-1":(-1, -0.5),
                "NN":(-0.5, 0.5),
                "W-1":(0.5, 1),
                "W-2":(1, 2),
                "W-3":(2, 3),
                "W-4":(3, 4)
                },
    
    'smds': {   
              "W-4":(0, 0.02),
              "W-3":(0.02, 0.05),
              "W-2":(0.05, 0.10),
              "W-1":(0.10, 0.20),
              "NN":(0.20, 0.80),
              "D-1":(0.80, 0.90),
              "D-2":(0.90, 0.95),
              "D-3":(0.95, 0.98),
              "D-4":(0.98, 1)
                 
            },
    
    'smci': { "D-4":(0, 0.02),
              "D-3":(0.02, 0.05),
              "D-2":(0.05, 0.10),
              "D-1":(0.10, 0.20),
              "NN":(0.20, 0.80),
              "W-1":(0.80, 0.90),
              "W-2":(0.90, 0.95),
              "W-3":(0.95, 0.98),
              "W-4":(0.98, 1)
    }
    
    
    }