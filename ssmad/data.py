import numpy as np
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
        

def read_grid_point_example(loc,
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

ascat_path = Path("/home/m294/ASCAT/081_ssm_userformat/datasets")

# Botswana
lat = -22.372
lon = 23.182
loc = (lon, lat)
ascat_ds = read_grid_point_example(loc , ascat_path)
ascat_ts = ascat_ds.get("ascat_ts")
ascat_gpi = ascat_ds.get("ascat_gpi")
sm_ts= ascat_ts.get("sm")
sm_ts.dropna(inplace=True)

print(type(ascat_ds))
print(ascat_ds.keys())