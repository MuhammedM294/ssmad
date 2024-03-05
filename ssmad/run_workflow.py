"""
run_workflow.py - SSMAD Workflow Execution

"""
__author__ = "Muhammed Abdelaal"
__email__ = "muhammedaabdelaal@gmail.com"

import pandas as pd
from pathlib import Path
from ssmad.data_reader import extract_obs_ts
from ssmad.anomaly import Z_Score


if __name__ == "__main__":
    
    # Example usage
  
    ascat_path = Path("/home/m294/VSA/Code/datasets")
    
    # Single point Morocco
    lat = 33.201
    lon = -7.373
    # loc  = 3597650
    # loc = (lon, lat)
    # sm_ts =  extract_obs_ts(loc, ascat_path, obs_type="sm" , read_bulk=False)["ts"]
    # df = pd.DataFrame(sm_ts)
    # #Compute monthly-based anomalies for October 2022 using z-score method
    # anomaly_df = Z_Score(df = df , 
    #                      variable='sm', 
    #                      time_step='month').detect_anomaly(year = 2022, month = 10)
    # print(anomaly_df)


    # multiple grid data points list in Austria 
    pointlist_path = "/home/m294/VSA/Code/pointlist_fibgrid_n6600000.csv" 
    pointlist = pd.read_csv(pointlist_path)
    pointlist = pointlist[:20]
    # Test anomalies computation for multiple points
    def compute_anomaly(gpi):
        sm_ts =  extract_obs_ts(gpi, ascat_path, obs_type="sm" , read_bulk=False)["ts"]
        df = pd.DataFrame(sm_ts)
        anomaly_df = Z_Score(df = df , 
                            variable='sm', 
                            time_step='month').detect_anomaly(year = 2022, month = 10)
        return anomaly_df['z_score'].values[0]
    
    from time import time
    start_time = time()
    pointlist['z_score'] = pointlist['point'].apply(lambda x: compute_anomaly(x))
    print("Time taken: ", time() - start_time)
    print(pointlist)
    
        
    