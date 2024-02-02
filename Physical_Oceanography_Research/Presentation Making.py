#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import statistics as stats
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
from scipy import optimize
import math
import netCDF4 as nc
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import ssl
import cartopy.feature as cfeature
import gsw
import datetime
import itertools as it
import scipy as sc
import haversine as hs
import random as rn
import matplotlib.patches as mpatches
import mpld3
from bokeh.plotting import output_file, show
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import plotly.graph_objects as go
import plotly.subplots as sp
import math
from scipy.stats import t
import json 
import xarray as xr
from scipy.interpolate import griddata


# # The next few code blocks are binning and giving bin indices for each bin.

# In[26]:


def read_in():

    saved_file1 = 'ship_calc_quantities.txt'
    with open(saved_file1, 'rb') as f1:
        dict_saved1 = pickle.load(f1)

    saved_file = 'float_calc_quantities.txt'
    with open(saved_file, 'rb') as f:
        dict_saved = pickle.load(f)
    merged_dict = dict_saved1 | dict_saved
    
    return merged_dict


# In[27]:


min_lat = -69
max_lat = -60
min_long = -146
max_long = -80
lat_bin_num = math.floor((max_lat-min_lat)/1.5)
long_bin_num = math.floor((max_long-min_long)/3.5)
temporal_bin_num = math.ceil(((2023-1992)*12 - 7)/2)


# In[28]:


temporal_bin_num


# In[29]:


dict_merged = read_in()
data = []
ids = []
lats = []
longs =[]
times = []
for float_id in dict_merged:
    lat = list(dict_merged[float_id]['LATITUDE'].values())
    long = list(dict_merged[float_id]['LONGITUDE'].values())
    julian_days = list(dict_merged[float_id]['JULIAN_TIME'].values())
    cycles = list(dict_merged[float_id]['LATITUDE'].keys())
    for i in range(len(lat)):
        if min_lat <= lat[i] <= max_lat and min_long <= long[i] <= max_long:
            data += [ [long[i], lat[i], julian_days[i]/365.25 + 1950] ]
            ids += [(float_id, cycles[i])]
            lats += [lat[i]]
            longs += [long[i]]
            times += [julian_days[i]/365.25 + 1950]


# In[30]:


data = np.array(data)
hist, edges = np.histogramdd(data, bins=[long_bin_num, lat_bin_num, temporal_bin_num])


# In[8]:


edges


# In[32]:


long_indices = np.digitize(np.array(longs), edges[0])
lat_indices = np.digitize(np.array(lats), edges[1])
time_indices = np.digitize(np.array(times), edges[2])
bin_indices = {}
for i in range(len(long_indices)):
    bin_indices[ids[i]] = (long_indices[i],lat_indices[i],time_indices[i])


# In[33]:


bin_indices


# # The next few code blocks are to determine average trends across depths in three different decades (1990, 2000, 2010) and plot them according to their decades.

# In[9]:


saved_file = 'float_trend_calc.txt'
#saved_file = 'float_trend_1k_calc.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[22]:


dict_saved['5906832'].keys()


# In[25]:


dict_saved['5906832'][19]['pres']


# In[9]:


def student90(num):
    n = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,50,60,80,100,120,np.inf]
    t = [6.3140,2.9200,2.3530,2.1320,2.0150,1.9430,1.8950,1.8600,1.8330,1.8120,1.7960,1.7820,1.7710,1.7610,1.7530,1.7460,1.7400,1.7340,1.7290,1.7250,1.7210,1.7170,1.7140,1.7110,1.7080,1.7060,1.7030,1.7010,1.6990,1.6970,1.6840,1.6760,1.6710,1.6640,1.6600,1.6580,1.6450]

    stud = float('nan')
    if num >= 1:
        idx = next((i for i, val in enumerate(n) if val >= num), len(n)-1)
        if idx > 0:
            stud = t[idx-1] + (t[idx] - t[idx-1]) * (num - n[idx-1]) / (n[idx] - n[idx-1])

    return stud


# In[10]:


dict_pres = {}
dict_pres[1990] = {}
dict_pres[2000] = {}
dict_pres[2010] = {}
pres_val = 1000

for float_id in dict_saved:
    
    dict_pres[1990][float_id] = {}
    dict_pres[2000][float_id] = {}
    dict_pres[2010][float_id] = {}
    
    for i in range(pres_val,5000, 10):

        dict_pres[1990][float_id][i] = {}
        dict_pres[2000][float_id][i] = {}
        dict_pres[2010][float_id][i] = {}
        
        dict_pres[1990][float_id][i]['trend'] = []
        dict_pres[2000][float_id][i]['trend'] = []
        dict_pres[2010][float_id][i]['trend'] = []
        
        dict_pres[1990][float_id][i]['ids'] = []
        dict_pres[2000][float_id][i]['ids'] = []
        dict_pres[2010][float_id][i]['ids'] = []

for float_id in dict_saved: 
    id_dict = {}
    for cycle_num in dict_saved[float_id]:
        pres = dict_saved[float_id][cycle_num]['pres']
        trend = dict_saved[float_id][cycle_num]['trend']
        ids = [yr for yr in dict_saved[float_id][cycle_num]['float_cycle']]
        year = [yr[0][:4] for yr in dict_saved[float_id][cycle_num]['float_cycle']]
#         julian_days = [day for day in dict_saved[float_id][cycle_num]['julian_time']] 
        
        for i in range(len(pres)):
            if year[i]== '2011' or year[i]== '2007':
                data_yr = 2000
            elif year[i]== '2016' or year[i]== '2018' or year[i]== '2017':
                data_yr = 2010
            else:
                data_yr = 1990
            for j in range(len(pres[i])): 
                dict_pres[data_yr][float_id][int(pres[i][j])]['trend'] += [trend[i][j]]
                dict_pres[data_yr][float_id][int(pres[i][j])]['ids'] += [ids[i]]
                
      
avg_dict = {} 

for years in dict_pres:
    avg_dict[years] = {}
    
    for i in range(1000, 5000, 10):
        avg_dict[years][i] = {}
        avg_dict[years][i]['trend'] = []
        avg_dict[years][i]['df'] = set()
       
    for float_id in dict_pres[years]:
        for pressure in dict_pres[years][float_id]:
            if len(dict_pres[years][float_id][pressure]['trend']) > 1:
                avg_dict[years][pressure]['trend'] += dict_pres[years][float_id][pressure]['trend']
                bin_ids = set([bin_indices[ids] for ids in dict_pres[years][float_id][pressure]['ids']])
                avg_dict[years][pressure]['df'].update( bin_ids )
                

    for i in range(1000, 5000, 10): 
        if len(avg_dict[years][i]['trend']) > 1:
            avg_dict[years][i]['mean_trend'] = np.mean(avg_dict[years][i]['trend'])*1000
            
            t_multiplier = student90(len(avg_dict[years][i]['df']))
            std_dev = stats.stdev(avg_dict[years][i]['trend'])
            avg_dict[years][i]['std_trend'] = t_multiplier*std_dev/np.sqrt(len(avg_dict[years][i]['df']))*1000


# In[ ]:


color_dict = {1990:'magenta', 2000: 'turquoise', 2010:'orange'}

plt.figure()
for year in avg_dict:
    Pressure = []
    Trend = []
    Std_er = []

    for pressure in avg_dict[year]:
        if 'mean_trend' in avg_dict[year][pressure]:
            Pressure.append(pressure)
            Trend.append(avg_dict[year][pressure]['mean_trend'])
            Std_er.append(avg_dict[year][pressure]['std_trend'])

    plt.plot(Trend, Pressure, color=color_dict[year], linewidth=3, label=year)
    
    # Use fill_betweenx instead of fill_between
    plt.fill_betweenx(Pressure, np.array(Trend)-np.array(Std_er), np.array(Trend)+np.array(Std_er), color=color_dict[year], alpha=0.25) # Adjust alpha for transparency

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.axvline(0, color='black', linewidth=0.75)
plt.gca().invert_yaxis()
plt.grid(True, color='gray', linestyle='--', linewidth=0.1)
plt.xlim(-5,20)
plt.ylabel('P  [dbar]', fontsize=30)
plt.xlabel(r'$\langle \partial C_T / \partial t \rangle$', fontsize=35)
label1 = '[m°C/yr]'
plt.text(0.63, -0.1, label1, transform=plt.gca().transAxes, fontsize=25, va='center')


plt.legend(loc='center right', fontsize=20)
plt.rc('figure', figsize=(11, 8))
plt.savefig(f'avgtrend1.png', transparent=True)


# In[ ]:


for year in avg_dict:
    

        length = max(list(avg_dict[year].keys()))
        sum1 = []
        std = []
        n = 0
        for i in range(3500, length, 10):
            if 'mean_trend' in avg_dict[year][i]:
                sum1 += [avg_dict[year][i]['mean_trend']]
                std += [avg_dict[year][i]['std_trend']]

        indices = ~np.isnan(std)
        sum1 = np.array(sum1)
        std = np.array(std)
        sum_mean = sum1[indices]
        std_mean = std[indices]
        mean_trend = np.mean(sum_mean)
        mean_unc = np.mean(std_mean)
        
        print(mean_trend, mean_unc)


# # The next few code blocks involve plotting tracks of floats and shipboard stations, separately and then, combinedly.

# In[ ]:


saved_file = 'float_trend_calc.txt'
#saved_file = 'float_trend_1k_calc.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[ ]:


# getting latitudes and longitudes of specific shipboard tracks and sorting according to the station numbers
ship_board = {}

for float_id in dict_saved:
    for cycle in dict_saved[float_id]:
        ship_stations = dict_saved[float_id][cycle]['float_cycle']
        
        for i in range(len(ship_stations)):
            if ship_stations[i][0] not in ship_board:
                ship_board[ship_stations[i][0]] = {}
            if ship_stations[i][1] not in ship_board[ship_stations[i][0]]:
                ship_board[ship_stations[i][0]][ship_stations[i][1]] = dict_saved[float_id][cycle]['loc'][i]
                
for ship in ship_board:
    ship_board[ship] = dict(sorted(ship_board[ship].items()))  


# In[ ]:


# deleting some track points of a shipboards, which have few stations very far from the rest. Adding these few far stations under a new shipboard id.

ship = ship_board['1992_P17E:P19S']
keys = []
for key in ship:
    if key > 40:
        keys.append(key)
ship_board['1992_P17E:P19S1'] = {}
for key in keys:
    ship_board['1992_P17E:P19S1'][key] = ship_board['1992_P17E:P19S'][key]
    del ship_board['1992_P17E:P19S'][key]


# In[ ]:


# plotting tracks of only float cycles
min_lat = -69
max_lat = -60
min_long = -144
max_long = -83


# Create a figure and axes with a specific projection
fig, ax = plt.subplots()

lat_list = {}
long_list = {}

# color_dict = {}
# for float_id in dict_saved:
#     if float_id == '3902325':
#         color_dict[float_id] = 'blue'
#     elif float_id == '3902326':
#         color_dict[float_id] = 'brown'
#     elif float_id == '3902327':
#         color_dict[float_id] = 'gray'
#     elif float_id == '3902328':
#         color_dict[float_id] = 'purple'
#     elif float_id == '5906830':
#         color_dict[float_id] = 'cyan'
#     elif float_id == '5906831':
#         color_dict[float_id] = 'orange'
#     elif float_id == '5906832':
#         color_dict[float_id] = 'green'
#     else:
#         color_dict[float_id] = 'red'

# Creating an arrow map
for float_id in dict_saved:
    lat_list[float_id] = []
    long_list[float_id] = []
   
    for cycle in dict_saved[float_id]:
        
        lat_list[float_id].append(dict_saved[float_id][cycle]['own_loc'][1])
        long_list[float_id].append(dict_saved[float_id][cycle]['own_loc'][0])
  
for float_id in lat_list:
    lat = lat_list[float_id]
    long = long_list[float_id]
    for i in range(len(lat) - 1):
        ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True, head_width=.05)
    ax.scatter(long, lat, s=2, color='lightsalmon')
ax.scatter(long[-1], lat[-1], s=2, color='lightsalmon', label='2023')
# for float_id in lat_list:
#     ax.text(np.mean(list(long_list[float_id])), max(list(lat_list[float_id]))+0.5, str(float_id), color=color_dict[float_id])

#     for i in range(len(lat) - 1):
#         if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
#             ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
#                  head_width=.05)
    # Uncomment the following line to include a legend for each float
    #ax.legend(f'Float {float_id}')

# for ship_id in dict_saved1:
#     lat = list(dict_saved1[ship_id]['LATITUDE'].values())
 
#     long = list(dict_saved1[ship_id]['LONGITUDE'].values())

  
#     lat_list += lat 
  
#     long_list += long 
    

#     for i in range(len(long) - 1):
#         if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
#             ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
#                  head_width=.05, color='red')
     

# Set the extent of the map
ax.set_xlim(min_long, max_long)
ax.set_ylim(min_lat, max_lat)

# Set the tick marks and label fonts
ax.set_ylabel('Latitude', fontstyle='italic', fontsize=17)
ax.set_xlabel('Longitude', fontstyle='italic', fontsize=17)

plt.legend(loc='upper right')
# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(8, 6))

# Show the plot
plt.show()
plt.savefig('newtrack.png')


# In[ ]:


# plotting tracks of only shipboard stations

min_lat = -69
max_lat = -61
min_long = -144
max_long = -83


# Create a figure and axes with a specific projection
fig, ax = plt.subplots()

lat_list = {}
long_list = {}

color_dict = {}
label_dict = {}
for ship_id in ship_board:
 
    if ship_id[:4] == '1992':
        color_dict[ship_id] = 'mediumpurple'
        label_dict[ship_id] = '1992'
    elif ship_id[:4] == '1994':
        color_dict[ship_id] = 'lightsteelblue'
        label_dict[ship_id] = '1994'
    elif ship_id[:4] == '2007':
        color_dict[ship_id] = 'cyan'
        label_dict[ship_id] = '2007'
    elif ship_id[:4] == '2011':
        color_dict[ship_id] = 'lime'
        label_dict[ship_id] = '2011'
    elif ship_id[:4] == '2016' or ship_id[:4] == '2017':
        color_dict[ship_id] = 'goldenrod'
        label_dict[ship_id] = '2016'
    elif ship_id[:4] == '2018':
        color_dict[ship_id] = 'bisque'
        label_dict[ship_id] = '2018'

# Creating an arrow map
for float_id in ship_board:
    lat_list[float_id] = []
    long_list[float_id] = []
   
    for cycle in ship_board[float_id]:
        
        lat_list[float_id].append(ship_board[float_id][cycle][1])
        long_list[float_id].append(ship_board[float_id][cycle][0])

id_list = []        
for float_id in lat_list:
    s = False
    for ids in id_list:
        if ids[:4] == float_id[:4]:
            s = True
            break
            
    if s == False:
        id_list.append(float_id)
        
    
    lat = lat_list[float_id]
    long = long_list[float_id]
    for i in range(len(lat) - 1):
        ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.5, length_includes_head=True, head_width=0.2, color=color_dict[float_id])
    
# Set the extent of the map
ax.set_xlim(min_long, max_long)
ax.set_ylim(min_lat, max_lat)

# Set the tick marks and label fonts
ax.set_ylabel('Latitude', fontstyle='italic', fontsize=17)
ax.set_xlabel('Longitude', fontstyle='italic', fontsize=17)

id_list.remove('2017_P17E')
id_list.sort(key=lambda x: x[:4])
handles = [mpatches.Patch(color=color_dict[float_id], label=label_dict[float_id]) for float_id in id_list]
plt.legend(handles=handles, loc='upper center')
#plt.legend(loc='upper center')
# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(8, 6))

# Show the plot
plt.show()
plt.savefig('newtrack2.png')


# In[ ]:


# plotting tracks of shipboard stations and float cycles, combined.

min_lat = -69
max_lat = -61
min_long = -144
max_long = -83


# Create a figure and axes with a specific projection
fig, ax = plt.subplots()

lat_list = {}
long_list = {}

color_dict = {}
label_dict = {}
for ship_id in ship_board:
 
    if ship_id[:4] == '1992':
        color_dict[ship_id] = 'mediumpurple'
        label_dict[ship_id] = '1992'
    elif ship_id[:4] == '1994':
        color_dict[ship_id] = 'lightsteelblue'
        label_dict[ship_id] = '1994'
    elif ship_id[:4] == '2007':
        color_dict[ship_id] = 'cyan'
        label_dict[ship_id] = '2007'
    elif ship_id[:4] == '2011':
        color_dict[ship_id] = 'lime'
        label_dict[ship_id] = '2011'
    elif ship_id[:4] == '2016' or ship_id[:4] == '2017':
        color_dict[ship_id] = 'goldenrod'
        label_dict[ship_id] = '2016'
    elif ship_id[:4] == '2018':
        color_dict[ship_id] = 'bisque'
        label_dict[ship_id] = '2018'
        
for float_id in dict_saved:
    color_dict[float_id] = 'lightsalmon'
    label_dict[float_id] = '2023'

# Creating an arrow map
for float_id in ship_board:
    lat_list[float_id] = []
    long_list[float_id] = []
   
    for cycle in ship_board[float_id]:
        
        lat_list[float_id].append(ship_board[float_id][cycle][1])
        long_list[float_id].append(ship_board[float_id][cycle][0])

id_list = []        
for float_id in lat_list:
    s = False
    for ids in id_list:
        if ids[:4] == float_id[:4]:
            s = True
            break
            
    if s == False:
        id_list.append(float_id)
        
    
    lat = lat_list[float_id]
    long = long_list[float_id]
    #for i in range(len(lat) - 1):
        #ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.5, length_includes_head=True, head_width=0.2, color=color_dict[float_id])
    ax.scatter(long, lat, color=color_dict[float_id], marker='+')
    
for float_id in dict_saved:
    lat_list[float_id] = []
    long_list[float_id] = []
   
    for cycle in dict_saved[float_id]:
        
        lat_list[float_id].append(dict_saved[float_id][cycle]['own_loc'][1])
        long_list[float_id].append(dict_saved[float_id][cycle]['own_loc'][0])
  
for float_id in lat_list:
    lat = lat_list[float_id]
    long = long_list[float_id]
#     for i in range(len(lat) - 1):
#         ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True, head_width=.2, color=color_dict[float_id])
    ax.scatter(long, lat, color=color_dict[float_id], marker='+')
    
id_list.append(float_id)
# Set the extent of the map
ax.set_xlim(min_long, max_long)
ax.set_ylim(min_lat, max_lat)

# Set the tick marks and label fonts
ax.set_ylabel('Latitude', fontstyle='italic', fontsize=17)
ax.set_xlabel('Longitude', fontstyle='italic', fontsize=17)

id_list.remove('2017_P17E')
id_list.sort(key=lambda x: x[:4])
handles = [mpatches.Patch(color=color_dict[float_id], label=label_dict[float_id]) for float_id in id_list]
plt.legend(handles=handles, loc='upper center')
#plt.legend(loc='upper center')
# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(6, 6))

# Show the plot
plt.show()
plt.savefig('newtrack3.png')


# # The next few code blocks are for plotting P-C_T for floats and shipboards near a point.

# In[11]:


def read_in():

    saved_file1 = 'ship_calc_quantities.txt'
    with open(saved_file1, 'rb') as f1:
        dict_saved1 = pickle.load(f1)

    saved_file = 'float_calc_quantities.txt'
    with open(saved_file, 'rb') as f:
        dict_saved = pickle.load(f)
    merged_dict = dict_saved1 | dict_saved
    
    return merged_dict


# In[12]:


def selecting_cycles(lat, long, dict0, dist=160):
    '''chooses cycles or stations from dict0 within the bin (width and length of dist) 
    centered on given lat, long
    Parameters:
    lat; float
    long; float
    dict0; dictionary
    dist; bin length in km; float
    returns a list of tuples (float/ship id, cycle/station number)
    '''
    
    loc = ( lat, long )
    lat_up = hs.inverse_haversine(loc, dist, hs.Direction.NORTH) # location 160km north
    lat_down = hs.inverse_haversine(loc, dist, hs.Direction.SOUTH) # location 160km south
    long_east = hs.inverse_haversine(loc, dist, hs.Direction.EAST) # location 160km east
    long_west = hs.inverse_haversine(loc, dist, hs.Direction.WEST) # location 160km west
    
    profiles = []
    for float_id in dict0:
        # getting latitudes for each float/ship
        longitudes = dict0[float_id]['LONGITUDE']
        latitudes = dict0[float_id]['LATITUDE']
        for cycle in latitudes:
            # the bin of width of 160km in latitude and length of 160km in longitude
            if lat_down[0] <= latitudes[cycle] <= lat_up[0]:
                if long_west[1] <= longitudes[cycle] <= long_east[1]:
                        profiles.append((float_id, cycle))

                        
    return profiles


# In[13]:


def high_press_data(dict_saved, press_val):   
    '''deletes the cycles/stations in dict_saved where pressure data is less than press_val
    Parameters:
    dict_saved; dictionary
    press_val; float
    returns a dictionary without the selected cycles
    '''
    keys_to_delete = []

    for float_id in dict_saved:
        pres = dict_saved[float_id]['PRES_LINEAR']
       
        for cycle in pres:
            if not max(pres[cycle]) >= press_val: # lower than threshold pressure
                keys_to_delete.append((float_id, cycle))

    # Delete the items outside of the loop
    for float_id, cycle in keys_to_delete:
        del dict_saved[float_id]['PRES_LINEAR'][cycle]
        del dict_saved[float_id]['CT_LINEAR'][cycle]
        del dict_saved[float_id]['SA_LINEAR'][cycle]
        del dict_saved[float_id]['SIGMA4_LINEAR'][cycle]
        del dict_saved[float_id]['LATITUDE'][cycle]
        del dict_saved[float_id]['LONGITUDE'][cycle]
        del dict_saved[float_id]['TIME_GREGORIAN'][cycle]
        del dict_saved[float_id]['JULIAN_TIME'][cycle]
    
    return dict_saved


# In[14]:


def select_high_pres(dict0, pres_val):
    '''gets rid of all quantities data below the threshold pressure (pres_val)
    Parameters:
    dict_saved; dictionary
    press_val; float
    returns a dictionary with data only above pres_val
    '''
    for float_id in dict0:
        
        pres = dict0[float_id]['PRES_LINEAR']
        ct = dict0[float_id]['CT_LINEAR']
        sa =  dict0[float_id]['SA_LINEAR']
        sigma4 = dict0[float_id]['SIGMA4_LINEAR']
        
        for cycle in pres:
            pres_id = int(np.where( pres[cycle] == pres_val )[0][0])
            dict0[float_id]['PRES_LINEAR'][cycle] = pres[cycle][pres_id: ]
            dict0[float_id]['CT_LINEAR'][cycle] = ct[cycle][pres_id: ]
            dict0[float_id]['SA_LINEAR'][cycle] = sa[cycle][pres_id: ]
            dict0[float_id]['SIGMA4_LINEAR'][cycle] = sigma4[cycle][pres_id: ]
            
    return dict0 


# In[ ]:


def plotting(dict0):
    
    '''plots P-CT, P-SA, P-Sigma4, CT-SA
    '''
    
    global lat, long
    global proflist
    # initiating dictionaries to get color and label each float. Color depends mainly on 
    # the year of the collected data. I chose dictionary format to avoid multiple labels 
    # for cycles from the same float or shipboard. 
    color_dict = {}
    label_dict = {}
    
    for ship_id in dict0:
        
        if ship_id[0] == '3' or ship_id[0] == '5':
            color_dict[ship_id]= 'black'
            label_dict[ship_id] = '2023'
        if ship_id[:4] == '1992':
            color_dict[ship_id] = 'magenta'
            label_dict[ship_id] = '1992'
        elif ship_id[:4] == '1993':
            color_dict[ship_id[:4]] = 'orchid'
            label_dict[ship_id[:4]] = '1993'
        elif ship_id[:4] == '1994':
            color_dict[ship_id] = 'thistle'
            label_dict[ship_id] = '1994'
        elif ship_id[:4] == '2007' or ship_id[:4] == '2008':
            color_dict[ship_id] = 'cyan'
            label_dict[ship_id] = '2007'
        elif ship_id[:4] == '2011':
            color_dict[ship_id] = 'lime'
            label_dict[ship_id] = '2011'
        elif ship_id[:4] == '2016' or ship_id[:4] == '2017':
            color_dict[ship_id] = 'darkorange'
            label_dict[ship_id] = '2017'
        elif ship_id[:4] == '2018':
            color_dict[ship_id] = 'goldenrod'
            label_dict[ship_id] = '2018'

    plt.figure() 
    ax = plt.gca()
    id_list = []
    unique_labels = []  # List to store unique labels
    # initiating lists to catch sa and ct so that I can use them later for contour plotting
#     sa_list = [] 
    ct_dict = {}
    years = ['1992', '1994', '2007', '2011', '2016', '2018', '2023']
    for year in years:
        ct_dict[year] = {}
        for i in range(1000, 5000, 10):
            ct_dict[year][i] = []
    
    pres_list = [] # initiating list to catch pressure to get the minimum pressure for the 
    # pressure axis labelling after inverting
   

    for float_id, cycle in proflist:
        
        if float_id not in id_list:
            if float_id[0]=='3' or float_id[0]=='5':
                year = '2023'
                s = True
                for ids in id_list:
                    if ids[0] == '3' or ids[0] == '5':
                        s = False
                if s:  
                    id_list.append(float_id)
            elif float_id[:4]=='1992':
                year = '1992'
                s = True
                for ids in id_list:
                    if ids[:4] == '1992':
                        s = False
                if s:  
                    id_list.append(float_id)
            
            else:        
                year = float_id[:4]
                id_list.append(float_id)
        
        ct = dict0[float_id]['CT_LINEAR'][cycle]
        pres = dict0[float_id]['PRES_LINEAR'][cycle] 
        ax.plot(ct, pres, color=color_dict[float_id])
                

        if label_dict[float_id] not in unique_labels:
            unique_labels.append(label_dict[float_id])
    
#     # Create the legend using the unique labels
    handles = [mpatches.Patch(color=color_dict[float_id], label=label_dict[float_id]) for float_id in id_list]
    ax.legend(handles=handles, loc='lower center', fontsize=15, markerscale=3)
            

    ax.set_xlabel(r'$C_T$ [$^\circ$C]', fontsize=25, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('P [dbar]', fontsize=25, fontstyle='italic', fontweight='bold')
    #axs[0,0].set_ylim(min(pres), max(pres)) # using ylim to avoid unnecessary extra axis lines
    ax.invert_yaxis()  # Invert the y-axis
    #ax.spines['bottom'].set_position(('data', min(pres)))  # Move the x-axis on top
    #ax.xaxis.tick_top()  # Move x-ticks on top
    #ax.xaxis.set_label_position('top')  # Move x-label on top
    #ax.spines['top'].set_visible(False)  # Remove unnecessary line on x-axis
    ax.grid(True, color='gray', linestyle='--', linewidth=0.1)

#     axs[0,1].set_xlabel('SA (g/kg)', fontsize=10)
#     axs[0,1].invert_yaxis()  # Invert the y-axis
#     axs[0,1].spines['bottom'].set_position(('data', min(pres)))  # Move the x-axis on top
#     axs[0,1].xaxis.tick_top()  # Move x-ticks on top
#     axs[0,1].xaxis.set_label_position('top')  # Move x-label on top
#     axs[0,1].spines['top'].set_visible(False)  # Remove unnecessary line on x-axis

#     axs[1,0].set_xlabel('SA (g/kg)', fontsize=10)
#     axs[1,0].set_ylabel('$C_T$ ($^\circ$C)', fontsize=10)
#     S, T = np.meshgrid(sa_list, ct_list)
#     potential_den = gsw.sigma4(S, T)
#     contour = axs[1,0].contour(S, T, potential_den, colors='k')
#     #axs[1,0].clabel(contour, inline=True, inline_spacing=3, rightside_up=True, colors='k', fontsize=8)
#     # label_positions = [(sa_val, ct_val) for sa_val, ct_val in zip(sa[cycle], ct[cycle])]
#     # ax.clabel(contour, inline=True, fontsize=8, manual=label_positions)

#     axs[1,1].set_xlabel(r'$\sigma_4$ ($kgm^{-3}$)', fontsize=10)
#     axs[1,1].invert_yaxis() # trend of hydrographic profile: lowest pressure (sea level) at top

    #legend = plt.legend(loc='lower right', fontsize=15, markerscale=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.rc('figure', figsize=(10, 9))
    fig_name = f'newlat{lat}_long{long}_pres1k.png'
    plt.savefig(fig_name)
    


# In[31]:


def plotting_2panels(dict0):
    
    '''plots P-CT, P-SA, P-Sigma4, CT-SA
    '''
    
    global lat, long, avg_dict
    global proflist
    # initiating dictionaries to get color and label each float. Color depends mainly on 
    # the year of the collected data. I chose dictionary format to avoid multiple labels 
    # for cycles from the same float or shipboard. 
    color_dict = {}
    label_dict = {}
    
    for ship_id in dict0:
        
        if ship_id[0] == '3' or ship_id[0] == '5':
            color_dict[ship_id]= 'lightsalmon'
            label_dict[ship_id] = '2023'
        if ship_id[:4] == '1992':
            color_dict[ship_id] = 'magenta'
            label_dict[ship_id] = '1992'
        elif ship_id[:4] == '1993':
            color_dict[ship_id[:4]] = 'orchid'
            label_dict[ship_id[:4]] = '1993'
        elif ship_id[:4] == '1994':
            color_dict[ship_id] = 'thistle'
            label_dict[ship_id] = '1994'
        elif ship_id[:4] == '2007' or ship_id[:4] == '2008':
            color_dict[ship_id] = 'cyan'
            label_dict[ship_id] = '2007'
        elif ship_id[:4] == '2011':
            color_dict[ship_id] = 'lime'
            label_dict[ship_id] = '2011'
        elif ship_id[:4] == '2016' or ship_id[:4] == '2017':
            color_dict[ship_id] = 'goldenrod'
            label_dict[ship_id] = '2017'
        elif ship_id[:4] == '2018':
            color_dict[ship_id] = 'bisque'
            label_dict[ship_id] = '2018'
            
    # Define the aspect ratio for each subplot
    aspect_ratio = 4 / 3

    # Define the width of each subplot
    subplot_width = 4  # This can be adjusted as needed

    # Calculate the total figure width to maintain the aspect ratio for each subplot
    figure_width = subplot_width * 2  # Since there are two subplots side by side
    figure_height = subplot_width * aspect_ratio

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(figure_width, figure_height)) 
    
    id_list = []
    unique_labels = []  # List to store unique labels
    # initiating lists to catch sa and ct so that I can use them later for contour plotting
#     sa_list = [] 
    ct_dict = {}
    years = ['1992', '1994', '2007', '2011', '2016', '2017', '2023']
    for year in years:
        ct_dict[year] = {}
        for i in range(1000, 5000, 10):
            ct_dict[year][i] = []
    
    pres_list = [] # initiating list to catch pressure to get the minimum pressure for the 
    # pressure axis labelling after inverting
   

    for float_id, cycle in proflist:
        
        if float_id not in id_list:
            if float_id[0]=='3' or float_id[0]=='5':
                year = '2023'
                s = True
                for ids in id_list:
                    if ids[0] == '3' or ids[0] == '5':
                        s = False
                if s:  
                    id_list.append(float_id)
            elif float_id[:4]=='1992':
                year = '1992'
                s = True
                for ids in id_list:
                    if ids[:4] == '1992':
                        s = False
                if s:  
                    id_list.append(float_id)
            
            else:        
                year = float_id[:4]
                id_list.append(float_id)
        
        ct = dict0[float_id]['CT_LINEAR'][cycle]
        pres = dict0[float_id]['PRES_LINEAR'][cycle] 
        ax.scatter(ct, pres, color=color_dict[float_id], s = 0.1)
                

        if label_dict[float_id] not in unique_labels:
            unique_labels.append(label_dict[float_id])
    
#     # Create the legend using the unique labels
    handles = [mpatches.Patch(color=color_dict[float_id], label=label_dict[float_id]) for float_id in id_list]
    ax.legend(handles=handles, loc='lower center', fontsize=10, markerscale=3)
            

    ax.set_xlabel(r'$C_T$ [$^\circ$C]', fontsize=17.5)
    ax.set_ylabel('P [dbar]', fontsize=20)
    ax.set_ylim(1000, 5000) # using ylim to avoid unnecessary extra axis lines
    ax.invert_yaxis()  # Invert the y-axis
 
    ax.grid(True, color='gray', linestyle='--', linewidth=0.01)

    color_dict = {1990:'red', 2000: 'blue', 2010:'brown'}


    for year in avg_dict:
        Pressure = []
        Trend = []
        Std_er = []

        for pressure in avg_dict[year]:
            if 'mean_trend' in avg_dict[year][pressure]:
                Pressure.append(pressure)
                Trend.append(avg_dict[year][pressure]['mean_trend'])
                Std_er.append(avg_dict[year][pressure]['std_trend'])

        ax1.plot(Trend, Pressure, color=color_dict[year], linewidth=3, label=f'{year}s')

        # Use fill_betweenx instead of fill_between
        ax1.fill_betweenx(Pressure, np.array(Trend)-np.array(Std_er), np.array(Trend)+np.array(Std_er), color=color_dict[year], alpha=0.25) # Adjust alpha for transparency


    ax1.axvline(0, color='black', linewidth=0.75)
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.01)
    ax1.set_xlim(-5,20)
#     ax1.set_ylabel('P  [dbar]', fontsize=15)
    ax1.set_xlabel(r'$\langle \partial C_T / \partial t \rangle$ [m$^\circ$C/yr]', fontsize=17.5)
    ax1.set_ylim(1000, 5000)
    ax1.invert_yaxis()

    ax1.legend(loc='lower right', fontsize=10)



#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
    plt.rc('figure', figsize=(6, 7))
    fig_name = f'profile_lat{lat}_long{long}_pres1k.pdf'
    plt.savefig(fig_name)
    


# In[ ]:


lat, long = -67, -102
min_lat = -68
max_lat = -61
min_long = -143
max_long = -83
pres_val = 1000
merged_dict = read_in()
merged_dict_whp = high_press_data(merged_dict, pres_val)
proflist = selecting_cycles(lat, long, merged_dict_whp)
# plotting(merged_dict_whp)
merged_dict_ohp = select_high_pres(merged_dict_whp, pres_val)
#proflist = selecting_cycles(lat, long, merged_dict_ohp)
#plot_track(merged_dict_ohp)
plotting(merged_dict_ohp)
# merged_dict_ohp = select_high_pres(merged_dict_whp, pres_val)
# proflist = selecting_cycles(lat, long, merged_dict_ohp)


# In[35]:


lat, long = -67, -135
min_lat = -70
max_lat = -60
min_long = -150
max_long = -80
pres_val = 1000
merged_dict = read_in()
merged_dict_whp = high_press_data(merged_dict, pres_val)
merged_dict_ohp = select_high_pres(merged_dict_whp, pres_val)
proflist = selecting_cycles(lat, long, merged_dict_ohp)
plotting_2panels(merged_dict_ohp)


# # Partial attempts to add bathymetry.

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Create a Basemap instance with the desired projection and boundaries
map = Basemap(projection='merc', llcrnrlon=-110, llcrnrlat=-60, urcrnrlon=-60, urcrnrlat=-30)

# Draw coastline and water
map.drawcoastlines()
map.fillcontinents(color='lightgray', lake_color='white')
map.drawmapboundary(fill_color='lightblue')

# Add title and display the map
plt.title('Coastline and Water of Southeastern Pacific Ocean')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read bathymetric data
dataset = xr.open_dataset('bathy.nc')

# Access the longitude, latitude, and depth values
longitude = dataset['lon'].values
latitude = dataset['lat'].values
depth = dataset['elevation'].values

# Create a meshgrid of longitude and latitude
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Create a figure and an axes with a specified projection
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the map boundaries
ax.set_extent([-145, -80, -70, -60], crs=ccrs.PlateCarree())

for float_id in lat_list:
    lat = lat_list[float_id]
    long = long_list[float_id]
    for i in range(len(lat) - 1):
        ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True, head_width=.05)
    ax.scatter(long, lat, s=2, color='lightsalmon')
ax.scatter(long[-1], lat[-1], s=2, color='lightsalmon', label='2023')

# Add coastline and water features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.contour(lon_mesh, lat_mesh, depth, levels=20, cmap='viridis')
plt.colorbar(label='Depth (m)')
# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Coastline and Water of Southeastern Pacific Ocean')

# Display the map
plt.show()
plt.savefig('modtrak.png')


# In[ ]:


import matplotlib.pyplot as plt

# Read bathymetric data
dataset = xr.open_dataset('bathy.nc')

# Access the longitude, latitude, and depth values
longitude = dataset['lon'].values
latitude = dataset['lat'].values
depth = dataset['elevation'].values

# Create a meshgrid of longitude and latitude
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Create the contour plot
plt.figure(figsize=(10, 6))
plt.contour(lon_mesh, lat_mesh, depth, levels=20, cmap='viridis')
plt.colorbar(label='Depth (m)')
plt.title('Bathymetry Contour Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[ ]:


len(latitude)


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read bathymetric data
dataset = xr.open_dataset('bathy.nc')

# Access the longitude, latitude, and depth values
longitude = dataset['lon'].values
latitude = dataset['lat'].values
depth = dataset['elevation'].values

# Create a meshgrid of longitude and latitude
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Create a figure and an axes with a specified projection
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the map boundaries
ax.set_extent([-145, -80, -70, -60], crs=ccrs.PlateCarree())

# Plot bathymetry contour
contour = ax.contour(lon_mesh, lat_mesh, depth, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax, label='Depth (m)')

# Add coastline and water features
ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.OCEAN, color='lightblue')

# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Coastline and Bathymetry of Southeastern Pacific Ocean')

# Display the map
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create a figure and an axes with a specified projection
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the map boundaries
ax.set_extent([-110, -60, -60, -30], crs=ccrs.PlateCarree())

# Add coastline and water features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.OCEAN, color='lightblue')

# Add country borders and names
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

countries = cfeature.NaturalEarthFeature(category='cultural',
                                        name='admin_0_countries',
                                        scale='10m',
                                        facecolor='none',
                                        edgecolor='gray')

ax.add_feature(countries)

# Add country labels
for country in countries.geometries():
    country_name = country.attributes['NAME_LONG']
    centroid = country.centroid
    ax.text(centroid.x, centroid.y, country_name,
            transform=ccrs.PlateCarree(),
            fontsize=8, ha='center', va='center')

# Add more ocean features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), color='lightblue')
ax.add_feature(cfeature.OCEAN.with_scale('10m'), color='lightblue')

# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Coastline and Water of Southeastern Pacific Ocean')

# Display the map
plt.show()


# # Plotting average trend vs longitude

# In[ ]:


saved_file = 'float_trend_calc.txt'
#saved_file = 'float_trend_1k_calc.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[ ]:


def lat67_data(dict_saved):
    
    dict_pres = {}
    dict_pres[1990] = {}
    dict_pres[2000] = {}
    dict_pres[2010] = {}
    pres_val = 1000

    for float_id in dict_saved:

        dict_pres[1990][float_id] = {}
        dict_pres[2000][float_id] = {}
        dict_pres[2010][float_id] = {}
        
        for cycle in dict_saved[float_id]:
            if -68.5 <= dict_saved[float_id][cycle]['own_loc'][1] <= -65.5:
             
                dict_pres[1990][float_id][cycle] = {}
                dict_pres[2000][float_id][cycle] = {}
                dict_pres[2010][float_id][cycle] = {}

                for i in range(pres_val,5000, 10):

                    dict_pres[1990][float_id][cycle][i] = {}
                    dict_pres[2000][float_id][cycle][i] = {}
                    dict_pres[2010][float_id][cycle][i] = {}

                    dict_pres[1990][float_id][cycle][i]['trend'] = []
                    dict_pres[2000][float_id][cycle][i]['trend'] = []
                    dict_pres[2010][float_id][cycle][i]['trend'] = []

                    dict_pres[1990][float_id][cycle][i]['long'] = dict_saved[float_id][cycle]['own_loc'][0]
                    dict_pres[2000][float_id][cycle][i]['long'] = dict_saved[float_id][cycle]['own_loc'][0]
                    dict_pres[2010][float_id][cycle][i]['long'] = dict_saved[float_id][cycle]['own_loc'][0]
                    

    for float_id in dict_saved: 
        for cycle in dict_saved[float_id]:
            if -68.5 <= dict_saved[float_id][cycle]['own_loc'][1] <= -65.5:
                pres = dict_saved[float_id][cycle]['pres']
                trend = dict_saved[float_id][cycle]['trend']
                ids = [yr for yr in dict_saved[float_id][cycle]['float_cycle']]
                year = [yr[0][:4] for yr in dict_saved[float_id][cycle]['float_cycle']]

                for i in range(len(pres)):
                    if year[i]== '2011' or year[i]== '2007':
                        data_yr = 2000
                    elif year[i]== '2016' or year[i]== '2018' or year[i]== '2017':
                        data_yr = 2010
                    else:
                        data_yr = 1990
                    for j in range(len(pres[i])): 
                        dict_pres[data_yr][float_id][cycle][int(pres[i][j])]['trend'] += [trend[i][j]]

    return dict_pres


# In[ ]:


dict_pres = lat67_data(dict_saved)


# In[ ]:


longitudes = {}
trends = {}
pres = 4500
for year in dict_pres:
    longitudes[year] = []
    trends[year] = []
    for float_id in dict_pres[year]:
        for cycle in dict_pres[year][float_id]:
            for i in range(len(dict_pres[year][float_id][cycle][pres]['trend'])):
                longitudes[year].append(dict_pres[year][float_id][cycle][pres]['long'])
                trends[year].append( dict_pres[year][float_id][cycle][pres]['trend'][i] )


# In[ ]:


plt.figure()
colors = {1990:'red', 2000:'green', 2010:'blue'}
markers = {1990:'+', 2000:'x', 2010:'o'}

for year in trends:
    if year == 2010:
        plt.scatter(longitudes[year], trends[year], marker=markers[year], color=colors[year])
    
plt.xlabel('Longitudes [°East]', fontsize=20, fontstyle='italic', fontweight='bold')
plt.ylabel('Trend [m°C/yr]', fontsize=20, fontstyle='italic', fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True, color='gray', linestyle='--', linewidth=0.1)
plt.rc('figure', figsize=(10, 10))


# In[ ]:


saved_file1 = 'ship_calc_quantities.txt'
with open(saved_file1, 'rb') as f1:
    dict_saved1 = pickle.load(f1)


# In[ ]:


max_list = []
for cycle in dict_saved1['1992_S04P']['btm_depth']:
    max_list.append(dict_saved1['1992_S04P']['btm_depth'][cycle])
print(max(max_list))


# In[ ]:


cycle_list = []
for cycle in dict_saved1['1992_S04P']['PRES_LINEAR']:
    if max(dict_saved1['1992_S04P']['PRES_LINEAR'][cycle])>4000:
        cycle_list.append(cycle)
        
print(len(cycle_list))


# In[ ]:


pres_val = 4200

for ship_id in dict_saved1:
    if ship_id == '2011_S04P':
        longitudes = []
        ct_dict = {}

        ship_long = dict_saved1[ship_id]['LONGITUDE']
        ship_lat = dict_saved1[ship_id]['LATITUDE']
        for cycle in ship_long:
            if -150 <= ship_long[cycle] <= -75 and -68.5 <= ship_lat[cycle] <= 65.5:
                if max(dict_saved1[ship_id]['PRES_LINEAR'][cycle])>=pres_val:
                    
                    ct_dict[ship_long[cycle]] = {}
                    longitudes.append(ship_long[cycle])
                    pres_id1000 = np.where(dict_saved1[ship_id]['PRES_LINEAR'][cycle]==1000)[0][0]
                    pres_id4200 = np.where(dict_saved1[ship_id]['PRES_LINEAR'][cycle]==pres_val)[0][0]
                    
        
                    for i in range(pres_id1000, pres_id4200+1):
                        ct_dict[ship_long[cycle]][dict_saved1[ship_id]['PRES_LINEAR'][cycle][i]] = dict_saved1[ship_id]['CT_LINEAR'][cycle][i]
                    
                    
                
     


# In[ ]:


pres_list = range(1000, 4210, 10)

Ct = []
Longitude, Pressure = np.meshgrid(longitudes, pres_list)
for pres in pres_list:
    ct_list = []
    for long in longitudes:
        ct_list.append(ct_dict[long][pres])
        
    Ct += [ct_list]


# In[ ]:


plt.figure()
plt.contourf(Longitude, Pressure, Ct, cmap='viridis')
plt.colorbar()  # Add a colorbar
plt.gca().invert_yaxis()


# In[ ]:


pres_val = 4200

for ship_id in dict_saved1:
    if ship_id == '2011_S04P':
        longitudes = []
        ct_dict = {}
        pres_dict = {}
        ship_long = dict_saved1[ship_id]['LONGITUDE']
        ship_lat = dict_saved1[ship_id]['LATITUDE']
        btm_depth = {}
        for cycle in ship_long:
            if -150 <= ship_long[cycle] <= -75 and -68.5 <= ship_lat[cycle] <= 65.5:
                btm_depth[ship_long[cycle]] = dict_saved1[ship_id]['btm_depth'][cycle]
                if max(dict_saved1[ship_id]['PRES_LINEAR'][cycle])>=1000:
                    
                    ct_dict[ship_long[cycle]] = {}
                    
                    longitudes.append(ship_long[cycle])
                    pres_id1000 = np.where(dict_saved1[ship_id]['PRES_LINEAR'][cycle]==1000)[0][0]
                    pres_dict[ship_long[cycle]] = dict_saved1[ship_id]['PRES_LINEAR'][cycle][pres_id1000:]
                    
        
                    for i in range(pres_id1000, len(dict_saved1[ship_id]['PRES_LINEAR'][cycle])):
                        ct_dict[ship_long[cycle]][dict_saved1[ship_id]['PRES_LINEAR'][cycle][i]] = dict_saved1[ship_id]['CT_LINEAR'][cycle][i]
                    
                    
                
     


# In[ ]:


for long in ct_dict:
    max_pres = max(ct_dict[long].keys())
    if 3400 <= max_pres:
        


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have pres_dict and ct_dict as input
max_pres = gsw.p_from_z([-6500], -67)
pres_list = np.arange(1000, 5000, 10)
longitudes = list(pres_dict.keys())
# depth_list = -1*gsw.z_from_p(pres_list, -67)
# Create 2D grid of pressure and longitude
P, L = np.meshgrid(pres_list, longitudes)

Ct = np.full(P.shape, np.nan)  # Initialize with NaN

for i, long in enumerate(longitudes):
    for j, pres in enumerate(pres_list):
        if pres in ct_dict[long]:
            Ct[i, j] = ct_dict[long][pres]
            
Ct_interpolated = griddata((L[~np.isnan(Ct)], P[~np.isnan(Ct)]), Ct[~np.isnan(Ct)], (L, P), method='linear')

depth_long = list(btm_depth.keys())
depth_val = - np.array(list(btm_depth.values()))
pres_val = gsw.p_from_z(depth_val, -67)            
            
plt.figure()
plt.contourf(L, P, Ct_interpolated, cmap='Blues')
cbar = plt.colorbar()
cbar.set_label('Temperature (°C)', rotation=90, labelpad=15)
plt.plot(depth_long, pres_val, color='black')
plt.fill_between(depth_long, pres_val, max_pres[0], where=(pres_val <= max_pres[0]), color='black')

plt.xlabel('Longitude (°E)', fontsize= 22)
plt.ylabel('P (dbar)', fontsize= 22)
plt.ylim(1000, max_pres[0])
plt.gca().invert_yaxis()
plt.rc('figure', figsize=(7.5, 6.1))
plt.show()
plt.savefig('inter_test_contour2011.png', transparent=True)


# In[ ]:


Ct = np.full(P.shape, np.nan)  # Initialize with NaN

for i, long in enumerate(longitudes):
    for j, pres in enumerate(pres_list):
        if pres in ct_dict[long]:
            Ct[i, j] = ct_dict[long][pres]


# In[ ]:


btm_depth.keys()


# In[ ]:


len(longitudes)


# In[ ]:


depth_long = list(btm_depth.keys())
depth_val = - np.array(list(btm_depth.values()))
pres_val = gsw.p_from_z(depth_val, -67)   


# In[ ]:


max_pres[0]


# In[ ]:




