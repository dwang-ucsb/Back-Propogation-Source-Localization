import numpy as np
import glob
import obspy
import utm

'''Read data and load into obspy.Stream object'''
array = obspy.Stream()

# Filtering and windowing
ti = obspy.UTCDateTime("") # time of interest
array.filter(type="Bandpass", freqmin=1.0, freqmax = 5.0)
array.trim(ti-2.5, ti+5+2.5) # take window around time

# access coords with array[n].stats.coordinates dict
# access by keying latitude or longitude
from obspy.core.util import AttribDict

# import this
station_coords = {}

for tr in array:
    key = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
    if key in station_coords:
        lat, lon = station_coords[key]
        tr.stats.coordinates = AttribDict({
            'latitude': lat,
            'longitude': lon
        })
    else:
        print(f"No coordinates found for {key}")

#Converting station coords to UTM 
for tr in array:
    lat = tr.stats.coordinates["latitude"]
    lon = tr.stats.coordinates["longitude"]
    easting, northing, zone_n, zone_l = utm.from_latlon(lat, lon)
    tr.stats.coordinates["easting"] = easting
    tr.stats.coordinates["northing"] = northing
    tr.stats.utm_zone = f"{zone_n}{zone_l}"

#define search area (in utm)
eastingbounds = np.array([0, 0])
northingbounds = np.array([0, 0])

resolution = 1 # resolution of grid to search over, in meters

x_grid = np.arange(eastingbounds[0], eastingbounds[1], resolution)
y_grid = np.arange(northingbounds[0], northingbounds[1], resolution)

"Grid-search algorithm"

# RTM for event @ 2023-10-20T21:02:10 to 2023-10-20T21:02:15

c = 440 # m/s

window_dt = 5
sample_rate = 100 #Hz
sample_window = window_dt*sample_rate 

t_k = int((ti - array[0].stats.starttime)*sample_rate) # sample index for time_of_interest
t_kf = int(t_k + sample_window)

grid_max_stack = np.zeros((x_grid.size, y_grid.size))

# grid search!
for i in range(x_grid.size):
    x = x_grid[i]
    for j in range(y_grid.size):
        y = y_grid[j]
        stack = np.zeros(sample_window)
        # iterate through array elements
        for trace in array:
            stat_x = trace.stats.coordinates['easting']
            stat_y = trace.stats.coordinates['northing']
            distance = np.sqrt((x-stat_x)**2 + (y-stat_y)**2)
            tracedata = trace.data
            dt = distance/c
            dsample = int(dt*sample_rate) # int() truncates/rounds down
            stack += tracedata[t_k+dsample:t_kf+dsample]
        grid_max_stack[i, j] = stack.max()

row, col = np.unravel_index(np.argmax(grid_max_stack), grid_max_stack.shape)
src_locUTM = (x_grid[row], y_grid[col], zone_n, zone_l)
src_locLatLon = utm.to_latlon(x_grid[row], y_grid[col], zone_n, zone_l)

#storing things to plot
critvalue = grid_max_stack[row, col]*.999
indices = np.argwhere(grid_max_stack >= critvalue)
gt_crit_region = [[x_grid[i], y_grid[j]] for i, j in indices]
gt_crit_region = np.array(gt_crit_region)
np.savez('plotting/RTMplotdata.npz', region=gt_crit_region, src_loc=src_locUTM, eastingbounds=eastingbounds,northingbounds=northingbounds)
np.savez('plotting/RTMcontourdata.npz', gridstack=grid_max_stack, x_grid=x_grid,y_grid=y_grid)