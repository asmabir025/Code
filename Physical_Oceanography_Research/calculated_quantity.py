#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import ssl
import cartopy.feature as cfeature
# import Basemap as bs
import gsw
import datetime
import itertools as it
import scipy as sc
import haversine as hs
import random as rn


# In[2]:


saved_file = 'ship_raw_data.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[4]:


for float_id in dict_saved:
    
    dict_saved[float_id]['PRES_LINEAR'] = {}
    dict_saved[ float_id ][ 'CT_LINEAR' ] = {}
    dict_saved[ float_id ][ 'SA_LINEAR' ] = {}
    dict_saved[ float_id ][ 'SIGMA4_LINEAR' ] = {}
    
    sea_pres = dict_saved[float_id]['PRES']
    insitu_T = dict_saved[float_id]['TEMP']
    prac_sal = dict_saved[float_id]['PSAL_CORRECTED']
    lat = dict_saved[float_id][ 'LATITUDE' ]
    long = dict_saved[float_id]['LONGITUDE']
    time_gregorian = dict_saved[float_id]['TIME_GREGORIAN']
    julian_time = dict_saved[float_id]['JULIAN_TIME']
    
    for cycle in sea_pres:
        abs_sal = gsw.SA_from_SP( prac_sal[cycle], sea_pres[cycle], long[cycle], lat[cycle] )
        #dict_saved[float_id]['SA'][cycle] = abs_sal
        
        c_T = gsw.CT_from_t( abs_sal, insitu_T[cycle], sea_pres[cycle] )
        #dict_saved[float_id]['C_T'][cycle] = c_T
        
        sigma4 = gsw.sigma4( abs_sal, c_T )
        #dict_saved[float_id]['Sigma4'][cycle] = sigma4
        
        f_CT = sc.interpolate.interp1d( sea_pres[cycle], c_T )
        f_SA = sc.interpolate.interp1d( sea_pres[cycle], abs_sal )
        f_sigma = sc.interpolate.interp1d( sea_pres[cycle], sigma4  )
        
        pres_linear = np.arange( 100, max(sea_pres[cycle]), 10 )
        y_CT = f_CT(pres_linear)
        y_SA = f_SA(pres_linear)
        y_sigma = f_sigma( pres_linear )
        
        dict_saved[float_id]['PRES_LINEAR'][cycle] = pres_linear
        dict_saved[ float_id ][ 'CT_LINEAR' ][cycle] = y_CT
        dict_saved[ float_id ][ 'SA_LINEAR' ][cycle] = y_SA
        dict_saved[ float_id ][ 'SIGMA4_LINEAR' ][cycle] = y_sigma


# In[5]:


for float_id in dict_saved:
    
    del dict_saved[float_id]['PRES']
    del dict_saved[float_id]['TEMP']
    del dict_saved[float_id]['PSAL_CORRECTED']
    

        
    


# In[5]:


saved_file = 'float_calc_quantities.txt'
with open(saved_file, 'wb') as f:
    pickle.dump(dict_saved, f, pickle.HIGHEST_PROTOCOL)


# In[6]:


saved_file = 'ship_raw_data.txt'
with open(saved_file, 'rb') as f:
    dict_saved1 = pickle.load(f)


# In[7]:


dict_saved1.keys()


# In[8]:


for float_id in dict_saved1:
    
    dict_saved1[float_id]['PRES_LINEAR'] = {}
    dict_saved1[ float_id ][ 'CT_LINEAR' ] = {}
    dict_saved1[ float_id ][ 'SA_LINEAR' ] = {}
    dict_saved1[ float_id ][ 'SIGMA4_LINEAR' ] = {}
    
    sea_pres = dict_saved1[float_id]['PRES']
    insitu_T = dict_saved1[float_id]['TEMP']
    prac_sal = dict_saved1[float_id]['PSAL_CORRECTED']
    lat = dict_saved1[float_id][ 'LATITUDE' ]
    long = dict_saved1[float_id]['LONGITUDE']
    time_gregorian = dict_saved1[float_id]['TIME_GREGORIAN']
    julian_time = dict_saved1[float_id]['JULIAN_TIME']
    
    for cycle in sea_pres:
            abs_sal = gsw.SA_from_SP( prac_sal[cycle], sea_pres[cycle], long[cycle], lat[cycle] )
            #dict_saved1[float_id]['SA'][cycle] = abs_sal

            c_T = gsw.CT_from_t( abs_sal, insitu_T[cycle], sea_pres[cycle] )
            #dict_saved1[float_id]['C_T'][cycle] = c_T

            sigma4 = gsw.sigma4( abs_sal, c_T )
            #dict_saved1[float_id]['Sigma4'][cycle] = sigma4

            f_CT = sc.interpolate.interp1d( sea_pres[cycle], c_T )
            f_SA = sc.interpolate.interp1d( sea_pres[cycle], abs_sal )
            f_sigma = sc.interpolate.interp1d( sea_pres[cycle], sigma4  )

            pres_linear = np.arange( 100, max(sea_pres[cycle]), 10 )
            y_CT = f_CT(pres_linear)
            y_SA = f_SA(pres_linear)
            y_sigma = f_sigma( pres_linear )

            dict_saved1[float_id]['PRES_LINEAR'][cycle] = pres_linear
            dict_saved1[ float_id ][ 'CT_LINEAR' ][cycle] = y_CT
            dict_saved1[ float_id ][ 'SA_LINEAR' ][cycle] = y_SA
            dict_saved1[ float_id ][ 'SIGMA4_LINEAR' ][cycle] = y_sigma


# In[9]:


for float_id in dict_saved1:
    
    
    del dict_saved1[float_id]['PRES']
    del dict_saved1[float_id]['TEMP']
    del dict_saved1[float_id]['PSAL_CORRECTED']
    


# In[10]:


saved_file1 = 'ship_calc_quantities.txt'
with open(saved_file1, 'wb') as f:
    pickle.dump(dict_saved1, f, pickle.HIGHEST_PROTOCOL)


# In[11]:


for float_id in dict_saved1:
    pres = dict_saved1[float_id]['PRES']
    for cycle in pres:
        if len(pres[cycle])<100:
            print(float_id, cycle)


# In[ ]:


dict_saved1.keys()


# In[ ]:


dict_saved['1992_S04P']['LONGITUDE']


# In[ ]:


float_id = '3902326'
ct = dict_saved[float_id]['C_T']
sa = dict_saved[float_id]['SA']


# In[ ]:


cycle = 10
S, T = np.meshgrid(sa[cycle], ct[cycle])
potential_den = gsw.sigma4( sa[cycle], ct[cycle] )


# In[ ]:


cycle = 16  # cycle number

S, T = np.meshgrid(sa[cycle], ct[cycle])
potential_den = gsw.sigma4(S, T)

fig, ax = plt.subplots()
contour = ax.contour(S, T, potential_den, colors='k')
ax.plot(sa[cycle], ct[cycle], label=f'cycle no. = {cycle}', color='r')
# label_positions = [(sa_val, ct_val) for sa_val, ct_val in zip(sa[cycle], ct[cycle])]
# ax.clabel(contour, inline=True, fontsize=8, manual=label_positions)
ax.clabel(contour, inline=True, inline_spacing=3, rightside_up=True, colors='k', fontsize=8)
plt.title(f'Deep Argo Float {float_id}', fontdict={'weight': 'bold'})
#ax.set_ylim(min(ct[cycle]), max(ct[cycle]))
ax.set_xlabel('SA (g/kg)', fontsize=20, loc='right', fontdict={'style': 'italic'})
ax.set_ylabel('$C_T$ ($^\circ$C)', fontsize=20, fontdict={'style': 'italic'})
plt.rc('figure', figsize=(6, 6))
plt.legend(loc=3)
plt.savefig('PressurevsSal.png')
plt.show()


# In[ ]:


# Generate temperature and salinity data
temperature = np.linspace(0, 30, 100)
salinity = np.linspace(32, 37, 100)

# Create a grid of temperature and salinity values
T, S = np.meshgrid(temperature, salinity)

# Calculate potential density at the sea surface (0 dbar)
potential_density = gsw.rho_t_exact(S, T, 0)

# Plot temperature-salinity diagram
fig, ax = plt.subplots()
contour = ax.contour(T, S, potential_density, colors='k')
ax.clabel(contour, inline=True, fontsize=8)

ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Salinity')

plt.show()


# In[ ]:


len(dict_saved[float_id]['TIME'].keys())


# In[ ]:


dict_saved.keys()


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import ssl

# Set the SSL certificate file path
ssl._create_default_https_context = ssl._create_unverified_context

# Create a figure and axes with a specific projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(80, 10))

# Add high-resolution coastline feature
coastline_feature = cfeature.NaturalEarthFeature(
    category='physical',
    name='coastline',
    scale='10m',
    edgecolor='black',
    facecolor='lightblue'
)

# Add the coastline feature to the plot
ax.add_feature(coastline_feature)



# Set the tick locators and formatters
ax.xaxis.set_major_locator(cticker.MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(cticker.MaxNLocator(nbins=6))
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax.tick_params(axis='both', labelsize=5)

lat_list = []
long_list = []

# Creating an arrow map
for float_id in dict_saved:
    lat = list(dict_saved[float_id]['LATITUDE'].values())
 
    long = list(dict_saved[float_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(lat) - 1):
        if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
            ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05)
    # Uncomment the following line to include a legend for each float
    #ax.legend(f'Float {float_id}')
min_lat = min(lat_list)
max_lat = max( lat_list )
min_long = min( long_list )
max_long = max( long_list )
for ship_id in dict_saved1:
    lat = list(dict_saved1[ship_id]['LATITUDE'].values())
 
    long = list(dict_saved1[ship_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(long) - 1):
        if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
            ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05, color='red')
    
    

# Set the extent of the map
ax.set_xlim(min_long, max_long)
ax.set_ylim(min_lat-1, max_lat+1)

# Set the tick marks and label fonts
ax.set_ylabel('Latitude', fontdict={'weight': 'bold'}, fontsize=10)
ax.set_xlabel('Longitude', fontdict={'weight': 'bold'}, fontsize=10)

# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(6, 6))

# Show the plot
plt.show()
plt.savefig('track.png')


# In[ ]:


# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.mpl.ticker as cticker
# import ssl

# Set the SSL certificate file path
ssl._create_default_https_context = ssl._create_unverified_context

# Create a figure and axes with a specific projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Add high-resolution coastline feature
# coastline_feature = cfeature.NaturalEarthFeature(
#     category='physical',
#     name='coastline',
#     scale='10m',
#     edgecolor='black',
#     facecolor='lightblue'
# )

# # Add the coastline feature to the plot
# ax.add_feature(coastline_feature)



# Set the tick locators and formatters
# ax.xaxis.set_major_locator(cticker.MaxNLocator(nbins=6))
# ax.yaxis.set_major_locator(cticker.MaxNLocator(nbins=6))
# ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
# ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax.tick_params(axis='both', labelsize=5)

lat_list = []
long_list = []

# Creating an arrow map
for float_id in dict_saved:
    lat = list(dict_saved[float_id]['LATITUDE'].values())
 
    long = list(dict_saved[float_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(lat) - 1):
        ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05)
    # Uncomment the following line to include a legend for each float
    #ax.legend(f'Float {float_id}')

for ship_id in dict_saved1:
    lat = list(dict_saved1[ship_id]['LATITUDE'].values())
 
    long = list(dict_saved1[ship_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(long) - 1):
        if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
            ax.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05, color='red')
    
    
min_lat = min(lat_list)
max_lat = max( lat_list )
min_long = min( long_list )
max_long = max( long_list )
# Set the extent of the map
ax.set_xlim(min_long, max_long)
ax.set_ylim(min_lat-1, max_lat+1)

# Set the tick marks and label fonts
ax.set_ylabel('Latitude', fontdict={'weight': 'bold'}, fontsize=10)
ax.set_xlabel('Longitude', fontdict={'weight': 'bold'}, fontsize=10)

# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(20, 20))

# Show the plot
plt.show()
plt.savefig('track.png')


# In[ ]:


plt.figure()



lat_list = []
long_list = []

for float_id in dict_saved:
    lat = list(dict_saved[float_id]['LATITUDE'].values())
 
    long = list(dict_saved[float_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(lat) - 1):
        plt.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05)
        
min_lat = min( lat_list )
max_lat = max( lat_list )
min_long = min( long_list )
max_long = max( long_list )


for ship_id in dict_saved1:
    lat = list(dict_saved1[ship_id]['LATITUDE'].values())
 
    long = list(dict_saved1[ship_id]['LONGITUDE'].values())

  
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(long) - 1):
        if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
            plt.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05, color='red')
    
    

plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)

# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(6, 6))

# Show the plot
plt.show()
plt.savefig('track.png')


# In[ ]:


def binning(bins_x=1.5, bins_y=1.5):
    global min_lat
    global max_lat
    global min_long
    global max_long
    global dict_saved
    
    lat_list = np.concatenate([dict_saved[float_id]['LATITUDE'][cycle] for float_id in float_id_list for cycle in cycle_dict[float_id]])
    long_list = np.concatenate([dict_saved[float_id]['LONGITUDE'][cycle] for float_id in float_id_list for cycle in cycle_dict[float_id]])
   

    # Calculate the number of bins in each dimension based on the bin width
    bins_x = int((int(max_long) - int(min_long)) / bin_width_x) + 1
    bins_y = int((int(max_lat) - int(min_lat)) / bin_width_y) + 1

    # Perform 2D binning
    hist, x_edges, y_edges = np.histogram2d(long_list, lat_list, bins=(bins_x, bins_y))
    
    bin_list = tuple(zip(x_edges, y_edges))
    
    for i in range(len(hist)):
        for j in range(len(hist[i])):
            if hist[i][j] == 0:
                bin_list[ bins_y*i + j] = 0
    

    return bin_list
            
            


# In[ ]:


s = tuple(zip((1,2), (3,4)))


# In[ ]:


# Define the bin width in each dimension
bin_width_x = 1.5
bin_width_y = 1.5

# Calculate the number of bins in each dimension based on the bin width
bins_x = int((max_long - min_long) / bin_width_x) + 1
bins_y = int((max_lat - min_lat) / bin_width_y) + 1

# Perform 2D binning
hist, x_edges, y_edges = np.histogram2d(long_list, lat_list, bins=(bins_x, bins_y))

# Display the result
# print(hist)
# print(x_edges)
# print(y_edges)
bin_list = [(x_edges[i], y_edges[j]) for i in range(len(x_edges)) for j in range(len(y_edges)) ]
print( bin_list )
# for i in range(len(hist)):
#     for j in range(len(hist[i])):
#         if hist[i][j] == 0:
#             bin_list[ bins_y*i + j] = (0,0)
# # print(bin_list)
# zero_indices = np.where(bin_list == (0,0))[0]
# bin_list = np.delete(bin_list, zero_indices, axis=-1)


# print( bin_list )


# In[ ]:


240*12


# In[ ]:


sum_ = 0
for i in range(len(hist)):
    for j in range(len(hist[i])):
        sum_ += 1


# In[ ]:


dict_saved.keys()


# In[ ]:


dict_saved['3902325']['LONGITUDE']


# In[ ]:


for float_id in dict_saved:
    longitudes = dict_saved[float_id]['LONGITUDE']
    latitudes = dict_saved[float_id]['LATITUDE']
    float_perm = it.permutations(zip(longitudes, latitudes))
    print(float_id, list(float_perm))


# In[ ]:


dict_saved1.keys()


# In[ ]:


for ship_id in dict_saved1:
    longitudes = dict_saved1[ship_id]['LONGITUDE'].values()
    latitudes = dict_saved1[ship_id]['LATITUDE'].values()
    loc = tuple(zip(longitudes, latitudes))
    print(ship_id)
    for ele in loc:
        print(ele)


# In[ ]:


def is_mon_inc(arr):
    '''checks if an array is monotonically increasing
    '''
    for i in range(len(arr)-1):
        if arr[i] >= arr[i+1]:
            return False
    return True 


# In[ ]:


non_moninc = []

for float_id in dict_saved:
    pres = dict_saved[float_id]['PRES']
    for cycle in pres:
        if is_mon_inc(dict_saved[float_id]['PRES'][cycle]) == False:
            non_moninc.append( (float_id, cycle) )
        


# In[ ]:


non_moninc


# In[ ]:


arr = dict_saved['5906830']['PRES'][12]
for i in range(len(arr)-1):
    if arr[i] > arr[i+1]:
        print( i )


# In[ ]:


import scipy as sc
x = dict_saved['3902325']['PRES'][33]
y = dict_saved['3902325']['C_T'][33]
f = sc.interpolate.interp1d( x, y )

x_new = np.arange( min(x), max(x), 10 )
y_new = f(x_new)


# In[ ]:


plt.figure()
plt.scatter(y, x, c = 'red', s=1)
plt.scatter(y_new, x_new, c='blue', s=1)
plt.gca().invert_yaxis()
plt.xlabel('$C_T$')
plt.ylabel('P')


# In[ ]:


profiles = []
for float_id in dict_saved:
    longitudes = dict_saved[float_id]['LONGITUDE']
    latitudes = dict_saved[float_id]['LATITUDE']
    for cycle in latitudes:
         if -68 <latitudes[cycle] < -66:
            if -110 <= longitudes[cycle] <= -95:
                    profiles.append((float_id, cycle))
profiles1 = []
for float_id in dict_saved1:
    longitudes = dict_saved1[float_id]['LONGITUDE']
    latitudes = dict_saved1[float_id]['LATITUDE']
    for cycle in latitudes:
         if -68 <latitudes[cycle] < -66:
            if -110 <= longitudes[cycle] <= -95:
                    profiles1.append((float_id, cycle))
                    
                    


# In[ ]:


saved_file = 'ship_calc_quantities.txt'
with open(saved_file, 'rb') as f:
    dict_saved1 = pickle.load(f)
saved_file = 'float_calc_quantities.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[ ]:


def selecting_cycles(lat, long, delta_lat=1.5, delta_long=3.75, dict0=dict_saved, dict1=dict_saved1):
    
    profiles = []
    for float_id in dict0:
        longitudes = dict0[float_id]['LONGITUDE']
        latitudes = dict0[float_id]['LATITUDE']
        for cycle in latitudes:
             if lat - delta_lat <= latitudes[cycle] <= lat + delta_lat:
                if long - delta_long <= longitudes[cycle] <= long + delta_long:
                        profiles.append((float_id, cycle))
    profiles1 = []
    for float_id in dict1:
        longitudes = dict1[float_id]['LONGITUDE']
        latitudes = dict1[float_id]['LATITUDE']
        for cycle in latitudes:
             if lat - delta_lat <= latitudes[cycle] <= lat + delta_lat:
                if long - delta_long <= longitudes[cycle] <= long + delta_long:
                        profiles1.append((float_id, cycle))
                        
    return profiles+profiles1


# In[ ]:


def high_press_data(dict_saved, press_val):   
    
    keys_to_delete = []

    for float_id in dict_saved:
        pres = dict_saved[float_id]['PRES_LINEAR']
       

        for cycle in pres:
            if not max(pres[cycle]) >= press_val:
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


# In[ ]:


def select_high_pres(dict0, pres_val):
    for float_id in dict0:
        
        pres = dict0[float_id]['PRES_LINEAR']
        ct = dict0[float_id]['CT_LINEAR']
        sa =  dict0[float_id]['SA_LINEAR']
        sigma4 = dict0[float_id]['SIGMA4_LINEAR']
        lat = dict0[float_id]['LATITUDE']
        long = dict0[float_id]['LONGITUDE']
        time_greg = dict0[float_id]['TIME_GREGORIAN']
        julian_time = dict0[float_id]['JULIAN_TIME']
        
        for cycle in pres:
            pres_id = int(np.where( pres[cycle] == pres_val )[0][0])
            dict0[float_id]['PRES_LINEAR'][cycle] = pres[cycle][pres_id: ]
            dict0[float_id]['CT_LINEAR'][cycle] = ct[cycle][pres_id: ]
            dict0[float_id]['SA_LINEAR'][cycle] = sa[cycle][pres_id: ]
            dict0[float_id]['SIGMA4_LINEAR'][cycle] = sigma4[cycle][pres_id: ]
            
    return dict0 


# In[ ]:


proflist = selecting_cycles(-67, -105)
dict_saved, dict_saved1 = high_press_data(dict_saved, 3000), high_press_data(dict_saved1, 3000)
dict_saved, dict_saved1 = select_high_pres(dict_saved, 3000), select_high_pres(dict_saved1, 3000)


# In[ ]:


def plotting(proflist, dict0=dict_saved, dict1=dict_saved1):

    color_dict = {}
    label_dict = {}
    for float_id in dict0:
        color_dict[float_id]= 'lightsalmon'
        label_dict[float_id] = '2023'
    for ship_id in dict1:
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
        elif ship_id[:4] == '2016':
            color_dict[ship_id] = 'goldenrod'
            label_dict[ship_id] = '2016'
        elif ship_id[:4] == '2018':
            color_dict[ship_id] = 'bisque'
            label_dict[ship_id] = '2018'

    fig, axs = plt.subplots(2, 2)        
#     id_list = []
    sa_list = []
    ct_list = []
    pres_list = []
    unique_labels = []  # List to store unique labels

    for float_id, cycle in proflist:


        if float_id[0] == '3':   
            if cycle in dict0[float_id]['CT_LINEAR']:
                ct = dict0[float_id]['CT_LINEAR'][cycle]
                pres = dict0[float_id]['PRES_LINEAR'][cycle] 
                sa = dict0[float_id]['SA_LINEAR'][cycle] 
                sigma4 = dict0[float_id]['SIGMA4_LINEAR'][cycle]
            else:
                continue

        else:
            if cycle in dict1[float_id]['CT_LINEAR']:
                ct = dict1[float_id]['CT_LINEAR'][cycle]
                pres = dict1[float_id]['PRES_LINEAR'][cycle] 
                sa = dict1[float_id]['SA_LINEAR'][cycle] 
                sigma4 = dict1[float_id]['SIGMA4_LINEAR'][cycle]
            else:
                continue



        scatter1 = axs[0,0].plot(ct, pres, color=color_dict[float_id], label=label_dict[float_id]) # plotting just one cycle            
        scatter2 = axs[0,1].plot(sa, pres, color=color_dict[float_id], label=label_dict[float_id]) # plotting just one cycle
        scatter3 = axs[1,0].plot(sa, ct, color=color_dict[float_id], label=label_dict[float_id])
        scatter4 = axs[1,1].plot(sigma4, pres, color=color_dict[float_id], label=label_dict[float_id]) # plotting just one cycle

        
        sa_list+= list(sa)
        ct_list+= list(ct)
        
        pres_list.append(pres)

#         if float_id not in id_list:
#             if float_id[0]=='3':
#                 for ids in id_list:
#                     if ids[0] == '3':
#                         id_list.append(float_id)
#                         break
#         if float_id not in id_list:
        if label_dict[float_id] not in unique_labels:
            unique_labels.append(label_dict[float_id])
            
#             id_list.append(float_id)
    legend1 = axs[0, 0].legend(handles=[scatter1], labels=unique_labels, loc='lower right')
    axs[0, 0].add_artist(legend1)

    axs[0,0].set_xlabel(r'$C_T$ ($^\circ$C)', fontsize=10)
    axs[0,0].set_ylabel('P (dbar)', fontsize=10)
    axs[0,0].set_ylim(min(pres), max(pres)) # using ylim to avoid unnecessary extra axis lines
    axs[0, 0].invert_yaxis()  # Invert the y-axis
    axs[0, 0].spines['bottom'].set_position(('data', min(pres)))  # Move the x-axis on top
    axs[0, 0].xaxis.tick_top()  # Move x-ticks on top
    axs[0, 0].xaxis.set_label_position('top')  # Move x-label on top
    axs[0, 0].spines['top'].set_visible(False)  # Remove unnecessary line on x-axis

    axs[0,1].set_xlabel('SA (g/kg)', fontsize=10)
    axs[0, 1].invert_yaxis()  # Invert the y-axis
    axs[0, 1].spines['bottom'].set_position(('data', min(pres)))  # Move the x-axis on top
    axs[0, 1].xaxis.tick_top()  # Move x-ticks on top
    axs[0, 1].xaxis.set_label_position('top')  # Move x-label on top
    axs[0, 1].spines['top'].set_visible(False)  # Remove unnecessary line on x-axis

    axs[1,0].set_xlabel('SA (g/kg)', fontsize=10)
    axs[1,0].set_ylabel('$C_T$ ($^\circ$C)', fontsize=10)
    S, T = np.meshgrid(sa_list, ct_list)
    potential_den = gsw.sigma4(S, T)
    contour = axs[1,0].contour(S, T, potential_den, colors='k', s=0.5)
    #axs[1,0].clabel(contour, inline=True, inline_spacing=3, rightside_up=True, colors='k', fontsize=8)
    # label_positions = [(sa_val, ct_val) for sa_val, ct_val in zip(sa[cycle], ct[cycle])]
    # ax.clabel(contour, inline=True, fontsize=8, manual=label_positions)

    axs[1,1].set_xlabel(r'$\sigma_4$ ($kgm^{-3}$)', fontsize=10)
    axs[1,1].set_ylim(min(pres), max(pres))
    axs[1,1].invert_yaxis() # trend of hydrographic profile: lowest pressure (sea level) at top

    plt.legend()
    plt.rc('figure', figsize=(8, 8))

    plt.savefig('highpres.png')


# In[ ]:


plotting(proflist)


# In[ ]:


keys_to_delete = []

for float_id in dict_saved:
    pres = dict_saved[float_id]['PRES_LINEAR']
    temp = dict_saved[float_id]['CT_LINEAR']
    sa = dict_saved[float_id]['SA_LINEAR'
    sigma4 = dict_saved[float_id]['SIGMA4_LINEAR']

    for cycle in pres:
        if not max(pres[cycle]) >= 3000:
            keys_to_delete.append((float_id, cycle))

# Delete the items outside of the loop
for float_id, cycle in keys_to_delete:
    del dict_saved[float_id]['PRES_LINEAR'][cycle]
    del dict_saved[float_id]['CT_LINEAR'][cycle]
    del dict_saved[float_id]['SA_LINEAR'][cycle]
    del dict_saved[float_id]['SIGMA4_LINEAR'][cycle]


# In[ ]:


for float_id in dict_saved:
    print(dict_saved[float_id]['LONGITUDE'].values())


# In[ ]:


saved_file = 'ship_calc_quantities.txt'
with open(saved_file, 'rb') as f:
    dict_saved1 = pickle.load(f)
saved_file = 'float_calc_quantities.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[ ]:


keys_to_delete = []

for float_id in dict_saved:
    pres = dict_saved[float_id]['PRES_LINEAR']
    temp = dict_saved[float_id]['CT_LINEAR']
    sa = dict_saved[float_id]['SA_LINEAR']
    sigma4 = dict_saved[float_id]['SIGMA4_LINEAR']

    for cycle in pres:
        if not max(pres[cycle]) >= 3000:
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
    


# In[ ]:


keys_to_delete = []

for float_id in dict_saved1:
    pres = dict_saved1[float_id]['PRES_LINEAR']
    temp = dict_saved1[float_id]['CT_LINEAR']
    sa = dict_saved1[float_id]['SA_LINEAR']
    sigma4 = dict_saved1[float_id]['SIGMA4_LINEAR']

    for cycle in pres:
        if not max(pres[cycle]) >= 3000:
            keys_to_delete.append((float_id, cycle))

# Delete the items outside of the loop
for float_id, cycle in keys_to_delete:
    del dict_saved1[float_id]['PRES_LINEAR'][cycle]
    del dict_saved1[float_id]['CT_LINEAR'][cycle]
    del dict_saved1[float_id]['SA_LINEAR'][cycle]
    del dict_saved1[float_id]['SIGMA4_LINEAR'][cycle]
    del dict_saved1[float_id]['LATITUDE'][cycle]
    del dict_saved1[float_id]['LONGITUDE'][cycle]
    del dict_saved1[float_id]['TIME_GREGORIAN'][cycle]
    del dict_saved1[float_id]['JULIAN_TIME'][cycle]


# In[ ]:


plt.figure()

lat_list = []
long_list = []
floatcycle_ids = []
for float_id in dict_saved:
    lat = list(dict_saved[float_id]['LATITUDE'].values())
 
    long = list(dict_saved[float_id]['LONGITUDE'].values())

    floatcycle_ids.append( (float_id, rn.choice(list(dict_saved[float_id]['LATITUDE'].keys()) ) ) ) 
    lat_list += lat 
  
    long_list += long 
    

    for i in range(len(lat) - 1):
        plt.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05)
        
min_lat = min(lat_list)
max_lat = max( lat_list )
min_long = min( long_list )
max_long = max( long_list )

shipstation_ids = []
lat_list = []
long_list = []
for ship_id in dict_saved1:
    lat = list(dict_saved1[ship_id]['LATITUDE'].values())
    long = list(dict_saved1[ship_id]['LONGITUDE'].values())
     

    for i in range(len(long) - 1):
        if -143.5 <long[i] < -79.5 and -68.5 <lat[i] < -59.5:
            plt.arrow(long[i], lat[i], long[i + 1] - long[i], lat[i + 1] - lat[i], lw=0.2, length_includes_head=True,
                 head_width=.05, color='red')
            lat_list += lat 
  
            long_list += long
    
    shipstation_ids.append( (ship_id, lat.index(rn.choice(lat)) ))
  

for float_id, cycle in floatcycle_ids:
    
    lat = dict_saved[float_id]['LATITUDE'][cycle]
    long = dict_saved[float_id]['LONGITUDE'][cycle]
    loc = ( lat, long )
    north = hs.inverse_haversine(loc, 160, hs.Direction.NORTH)
    south = hs.inverse_haversine(loc, 160, hs.Direction.SOUTH)
    east = hs.inverse_haversine(loc, 160, hs.Direction.EAST)
    west = hs.inverse_haversine(loc, 160, hs.Direction.WEST)
    
    plt.arrow(long, lat, north[1] - long, north[0] - lat, lw=0.2, length_includes_head=True,
                 head_width=.2, color='blue')
    plt.arrow(long, lat, south[1] - long, south[0] - lat, lw=0.2, length_includes_head=True,
                 head_width=.2, color='blue')
    plt.arrow(long, lat, east[1] - long, east[0] - lat, lw=0.2, length_includes_head=True,
                 head_width=.2, color='blue')
    plt.arrow(long, lat, west[1] - long, west[0] - lat, lw=0.2, length_includes_head=True,
                 head_width=.2, color='blue')
     
    
# for float_id, cycle in shipstation_ids:
    
#     lat = dict_saved1[float_id]['LATITUDE'][cycle]
#     long = dict_saved1[float_id]['LONGITUDE'][cycle]
#     loc = ( lat, long )
#     north = hs.inverse_haversine(loc, 160, hs.Direction.NORTH)
#     south = hs.inverse_haversine(loc, 160, hs.Direction.SOUTH)
#     east = hs.inverse_haversine(loc, 160, hs.Direction.EAST)
#     west = hs.inverse_haversine(loc, 160, hs.Direction.WEST)
    
#     plt.arrow(long, lat, north[1] - long, north[0] - lat, lw=0.2, length_includes_head=True,
#                  head_width=.2, color='blue')
#     plt.arrow(long, lat, south[1] - long, south[0] - lat, lw=0.2, length_includes_head=True,
#                  head_width=.2, color='blue')
#     plt.arrow(long, lat, east[1] - long, east[0] - lat, lw=0.2, length_includes_head=True,
#                  head_width=.2, color='blue')
#     plt.arrow(long, lat, west[1] - long, west[0] - lat, lw=0.2, length_includes_head=True,
#                  head_width=.2, color='blue')
     
        
plt.xlim(min_long-1, max_long+1)
plt.ylim(min_lat-0.5, max_lat+0.5)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)

# Uncomment the following line to control the size of the figure
plt.rc('figure', figsize=(6, 8))

# Show the plot
plt.show()
plt.savefig('track.png')


# In[2]:


saved_file = 'float_calc_quantities.txt'
with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[7]:


for ar in dict_saved['3902327']['PRES_LINEAR']:
    print( dict_saved['3902327']['PRES_LINEAR'][ar][-1]  )


# In[ ]:




