"""
run_workflow.py - SSMAD Workflow Execution

"""
__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

import pandas as pd
from pathlib import Path
from ssmad.data_reader import*
from ssmad.anomaly import Z_Score , SMAPI
import concurrent.futures
from ssmad.utils import create_logger, log_exception, log_time



def set_path():

    # austria = "/home/m294/VSA/Code/pointlist_AUT.csv"
    Germany = "/home/m294/VSA/Code/pointlist_DEU.csv"
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    return ascat_path, Germany

def load_pointlist(aoi):
    pointlist = pd.read_csv(aoi)
    return pointlist

logger = create_logger('run_logger')

@log_exception(logger)
def run(gpi):
    
    ascat_ts = ascat_obj.read(gpi)
    sm_ts = ascat_ts.get("sm")
    df = pd.DataFrame(sm_ts)
    z_score = Z_Score(df = df , 
                        variable='sm', 
                        time_step='month').detect_anomaly(year = 2022, month = 10)

    
    return (gpi, z_score['z_score'].values[0] )
  

@log_time(logger)
def run_workflow():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run, pointlist['point'])
        for r in result:
            if r:
                pointlist.loc[pointlist['point'] == r[0], 'z_score'] = r[1]
                
                
    return pointlist


if __name__ == "__main__":
    
    # Example usage
    ascat_path, aoi = set_path()
    ascat_obj = AscatData(ascat_path, False)
    pointlist = load_pointlist(aoi)
    pointlist = pointlist[:100]
    df = run_workflow()
    print(df)
    # pointlist.to_csv('ssmad/DEU_zscore.csv') 
    