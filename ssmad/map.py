import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import eomaps
from metadata import indicators_thresholds
matplotlib.use('Qt5Agg') 


def set_thresholds(method):
    """
    Set the thresholds for the specified method based on the method name.
    
    parameters:
    -----------
    
    method: str
        The method name for which the thresholds are to be set. Supported methods are:
        'zscore', 'smapi', 'smdi', 'smca', 'smad', 'smci', 'smds', 'essmi', 'beta', 'gamma'
    """
    
    if method in ['beta', 'gamma' , 'essmi', 'zscore','smad']:
        return indicators_thresholds['zscore']
    else:
        return indicators_thresholds[method]
    
def set_extent(df, x='lon', y='lat', buffer=2):
    """
    Set the extent for the map based on the provided dataframe and buffer.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        The dataframe containing the data.
        
    x: str
        The column name for the x-axis.
        
    y: str
        The column name for the y-axis.
        
    buffer: int
        The buffer to be added to the min and max values of the x and y axis.
    """
    
    min_lat = df[y].min() - buffer
    max_lat = df[y].max() + buffer
    min_lon = df[x].min() - buffer
    max_lon = df[x].max() + buffer
    
    return min_lon, max_lon, min_lat, max_lat


def set_bins(colm):
    """
    Set the bins and labels for color classification for the selected column.
    
    parameters:
    -----------
    
    colm: str
        The data column name for which the bins and labels are to be set.
    """
    method = colm.split('(')[0]
    if "-" in method:
        method = method.split('-')[0]
        
    thrsholds = set_thresholds(method)
    bins = [val[1] for val in thrsholds.values()]
    labels = [key for key in thrsholds.keys()]
    labels.insert(0, labels[0])
    bins.insert(0, next(iter(thrsholds.values()))[0])
    
    return bins, labels

def plot_anomaly_map(df, colm, x = "lon", y="lat",
                     crs = 4326,  set_extent_to = True, extent=None,
                     figsize=(7,7), title = "SM Anomaly", 
                     bins =None, labels=None,
                     add_gridlines=True,
                     g_kwargs={'d': (2, 2), 'ec': 'grey', 'ls': '--', 'lw': 0.01,  },
                     vmin=None, vmax=None,
                     add_colorbar=True,
                     cmap='RdYlBu',
                     cb_kwargs={'pos':0.4, 'labelsize':7, 'tick_lines':'center', 'show_values':False}
                    ):
    
    """
    Plot the anomaly map for the selected column from the dataframe.
    
    parameters:
    -----------
    
    df: pd.DataFrame
        The dataframe containing the data.
        
    colm: str
        The column name for the data to be plotted.
        
    x: str
        The column name for the x-axis.
        
    y: str
        The column name for the y-axis.
        
    crs: int
        The coordinate reference system for the map.
        
    set_extent_to: bool
        Whether to set the map to a specific extent.
        
    extent: tuple
        The extent to set the map to. if None, the extent is calculated based on the data.
        
    figsize: tuple
        The figure size for the map.
        
    title: str
        The title for the map.
        
    bins: list
        The bins for the color classification.
        
    labels: list
        The labels for the bins of the color classification.
        
    add_gridlines: bool
        Whether to add gridlines to the map.
        
    g_kwargs: dict
        The gridlines keyword arguments. This includes the linestyle, linewidth, edgecolor, and distance between gridlines.
        
    vmin: float
        The minimum value for the color classification.
        
    vmax: float
        The maximum value for the color classification.
        
    cmap: str
        The colormap for the map.
        
    add_colorbar: bool
        Whether to add a colorbar to the map.
        
    cb_kwargs: dict
        The colorbar keyword arguments. This includes the position, label size, tick lines, and whether to show values.
        
    
    """
    
    # Initialize map object
    m = eomaps.Maps(figsize=figsize, frameon=True, crs=crs)
    m.ax.set_title(title, pad=20, linespacing=1.5)
    m.set_shape.rectangles(radius=0.05)
    m.set_frame(linewidth=0.5)
    
    # Set data and parameter
    m.set_data(data=df, parameter=colm, x=x, y=y, crs=crs)
    
    # Set extent
    if set_extent_to:
        extent = set_extent(df, x=x, y=y) if extent is None else extent
        m.ax.set_extent(extent, eomaps.Maps.CRS.PlateCarree())
    
    # Set bins and labels
    if bins is None and labels is None:
        bins, labels = set_bins(colm)
    
    # Set classification
    m.set_classify.UserDefined(bins=bins)
    # Add gridlines
    if add_gridlines:
        g = m.add_gridlines(**g_kwargs)
        g.add_labels(fontsize=8 )
    
    # Set vmin and vmax based on the thresholds min and max
    vmin = bins[0] if vmin is None else vmin
    vmax = bins[-1] if vmax is None else vmax
    
    # Plot map
    cmap = cmap+"_r" if colm.split('(')[0] == 'smds' else cmap
    m.plot_map(vmin=vmin, vmax=vmax, cmap=cmap, lw=1.5)
    
    # Add colorbar
    
    if add_colorbar:
        cb = m.add_colorbar(pos=cb_kwargs['pos'], )
        cb.tick_params(labelsize=cb_kwargs['labelsize'])
        cb.set_bin_labels(bins=bins, names=labels,
                          tick_lines=cb_kwargs['tick_lines'], 
                          show_values=cb_kwargs['show_values'])
    
    m.show()



if __name__ == "__main__":
    
    df = pd.read_csv('/home/m294/results.csv')
    colm = 'smdi(2021-7)'
    

    
    plot_anomaly_map(df, colm,set_extent_to=True ,title = "Gamma Anomaly Map", figsize=(10,8))
    


