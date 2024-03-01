import numpy as np
import pandas as pd
from pathlib import Path
from fibgrid.realization import FibGrid
from pynetcf.time_series import GriddedNcContiguousRaggedTs



class AscatData(GriddedNcContiguousRaggedTs):
    """
    Class reading ASCAT SSM 6.25 km data.
    """

    def __init__(self, path, read_bulk=True):
        """
        Initialize ASCAT data.

        Parameters
        ----------
        path : str
            Path to dataset.
        read_bulk : bool, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        """
        grid = FibGrid(6.25)
        ioclass_kws = dict(read_bulk=read_bulk)
        super().__init__(path, grid, ioclass_kws=ioclass_kws)
        

def read_grid_point(loc,
                   ascat_sm_path,
                   read_bulk=False):
    """
    Read grid point for given lon/lat coordinates or grid_point.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.
    ascat_sm_path : str
        Path to ASCAT soil moisture data.
    read_bulk : bool, optional
        If "True" all data will be read in memory, if "False"
        only a single time series is read (default: False).
        Use "True" to process multiple GPIs in a loop and "False" to
        read/analyze a single time series.
    """
    data = {}

    print(f"Reading ASCAT soil moisture: {ascat_sm_path}")
    ascat_obj = AscatData(ascat_sm_path, read_bulk)

    if isinstance(loc, tuple):
        lon, lat = loc
        ascat_gpi, distance = ascat_obj.grid.find_nearest_gpi(lon, lat)
        print(f"ASCAT GPI: {ascat_gpi} - distance: {distance:8.3f} m")
    else:
        ascat_gpi = loc
        lon, lat = ascat_obj.grid.gpi2lonlat(ascat_gpi)
        print(f"ASCAT GPI: {ascat_gpi}")

    ascat_ts = ascat_obj.read(ascat_gpi)

    if ascat_ts is None:
        raise RuntimeError(f"ASCAT soil moisture data not found: {ascat_sm_path}")

    # set observations to NaN with less then two observations
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan
    data["ascat_ts"] = ascat_ts
    data["ascat_gpi"] = ascat_gpi
    data["ascat_lon"] = lon
    data["ascat_lat"] = lat

    return data


def extract_obs_ts(loc, ascat_path, obs_type="sm", read_bulk=False):
    """
    Read time series of given observation type.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.
    ascat_path : str
        Path to ASCAT soil moisture data.
    obs : str, optional
        Observation type (default: "sm").
    read_bulk : bool, optional
        If "True" all data will be read in memory, if "False"
        only a single time series is read (default: False).
        Use "True" to process multiple GPIs in a loop and "False" to
        read/analyze a single time series.
    """
    data = read_grid_point(loc, ascat_path, read_bulk)
    ascat_ts = data.get("ascat_ts")
    lat = data.get("ascat_lat")
    lon = data.get("ascat_lon")
    gpi = data.get("ascat_gpi")
    ts= ascat_ts.get(obs_type)
    ts.dropna(inplace=True)
    
    return {"ts": pd.DataFrame(ts), "lon": lon, "lat": lat, "gpi": gpi}





if __name__ == "__main__":
    
      pass