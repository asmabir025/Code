#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
import pickle
import csv
import statistics as stats
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
from scipy import optimize
import urllib
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


# In[ ]:


def quality_data(   ):
    
    # getting masked_arrays of temperature, pressure, and salinity, and also the qc (quality control) arrays
    
    global mask_arT
    global mask_arP
    global mask_arS
    global qc_arT
    global qc_arP
    global qc_arS
    
    # indices of data arrays where mask = True
    invalid_id1 = np.where(mask_arT.mask) 
    invalid_id2 = np.where(mask_arP.mask)
    invalid_id3 = np.where(mask_arS.mask)
    
    invalid_id_list = [] # initiating a list of tuples (these tuples will contain just integer 'invalid' indices) 
    invalid_id_list += [ (invalid_id1[0][i], invalid_id1[1][i]) for i in range(len(invalid_id1[0])) ]  
    invalid_id_list += [ (invalid_id2[0][i], invalid_id2[1][i]) for i in range(len(invalid_id2[0])) ]
    invalid_id_list += [ (invalid_id3[0][i], invalid_id3[1][i]) for i in range(len(invalid_id3[0])) ]
    
    invalid_id_set = set(invalid_id_list) # to avoid repitition of same indices
    
   
    # indices of qc arrays where mask = False (unmasked) and qc value 1 or 2 (or 3 for salinity only)
    
    valid_id = np.where(np.logical_and(qc_arT.mask == False, qc_arT.data != b''))
    valid_id1 = [ (valid_id[0][i], valid_id[1][i]) for i in range(len(valid_id[0])) \
                 if int(qc_arT.data[valid_id[0][i]][valid_id[1][i]]) < 3 ]
    
    valid_id = np.where(np.logical_and(qc_arP.mask == False, qc_arP.data != b''))
    valid_id2 = [ (valid_id[0][i], valid_id[1][i]) for i in range(len(valid_id[0])) \
                 if int(qc_arP.data[valid_id[0][i]][valid_id[1][i]]) < 3 ]
    
    valid_id = np.where(np.logical_and(qc_arS.mask == False, qc_arS.data != b''))
    valid_id3 = [ (valid_id[0][i], valid_id[1][i]) for i in range(len(valid_id[0])) \
                 if int(qc_arS.data[valid_id[0][i]][valid_id[1][i]]) < 4 ]
    

    valid_id_set0 = set.intersection(set(valid_id1), set(valid_id2), set(valid_id3)) # A set of tuples (these tuples will contain just valid integer indices) 

    
    # masks of qc arrays (i.e.,ds['TEMP_QC']) and masks of corresponding data arrays (i.e.,ds['TEMP']) 
    # are essentially same. Just to be cautious, in case, the masks are not same, I subtracted the set of 
    # masked ids of data array from the valid id set.
    valid_id_set = valid_id_set0 - invalid_id_set

    # initiating a dictionary to catch valid values for T, P, and S
    dictT = {} 
    dictP = {}
    dictS = {} 
    
    for i in range(len(mask_arT)): # len(mask_array) = number of cycles
        dictT[ i ] = [] # iniitating an empty list for cycle i + 1; i+1 to ensure that first cycle no = 1  
        dictP[ i ] = []
        dictS[ i ] = []
        for j in range(len(mask_arT[i])): # len(mask_array[i]) = number of observations in a cycle
            if (i,j) in valid_id_set:
                dictT[i].append( mask_arT[i][j] )
                dictP[i].append( mask_arP[i][j] )
                dictS[i].append( mask_arS[i][j] )
    
                
    return dictT, dictP, dictS


# In[ ]:


def julian_time_to_utc(julian_year_start_utc, julian_time):
    # Extract year, month, day, hour, minute, second from Julian year start
    year = int(julian_year_start_utc[:4])
    month = int(julian_year_start_utc[4:6])
    day = int(julian_year_start_utc[6:8])
    hour = int(julian_year_start_utc[8:10])
    minute = int(julian_year_start_utc[10:12])
    second = int(julian_year_start_utc[12:14])

    # Convert Julian year start UTC to a datetime object
    dt_start_utc = datetime.datetime(year, month, day, hour, minute, second)

    # Calculate the datetime for the given Julian time
    dt = dt_start_utc + datetime.timedelta(days=julian_time)

    return dt


# In[ ]:


def reference_utc( reference_utc_ar ):
    '''converts array containing numpy bytes of reference time (utc) to utc time string
    '''
    julian_start_str = [ int(i) for i in reference_utc_ar ]

    julian_yr = ''

    for string in julian_start_str:
        julian_yr += str(string)
        
    return julian_yr


# In[ ]:


def S_new(s,t,p):
    # coeffiecients

    Tcor=3.25e-06
    CPcor_SBE=-9.57e-08
    CPcor_new=-12.5e-08

    
    # computing conductivity from dictionaries of practical salinity s, in-situ temperature t, and sea pressure p
    S_new = {}
    for cycle in s:
        co=gsw.C_from_SP(s[cycle],t[cycle],p[cycle])
        # backing out incorrect pressure coefficient and reapplying correct one
        co_new = []
        for i in range(len(t[cycle])):
            co_new += [ co[i]*(1+Tcor*t[cycle][i]+CPcor_SBE*p[cycle][i])/(1+Tcor*t[cycle][i]+CPcor_new*p[cycle][i]) ]
        # recalculating corrected sp_new from corrected co_new
        S_new[cycle]=gsw.SP_from_C(co_new,t[cycle],p[cycle])
    
    return S_new


# In[ ]:


# listing all Deep Argo file name strings 
file_list = [ '3902325_prof.nc', '3902326_prof.nc', '3902327_prof.nc','3902328_prof.nc', '5906830_prof.nc', '5906831_prof.nc', '5906832_prof.nc', '5906833_prof.nc']


# In[ ]:


saved_file = 'float_raw_data.txt'


# In[ ]:


# saving good quality data for each float

saved_dict = {}
for file in file_list:
    
    float_id = file[0:-8]
    saved_dict[float_id] = {}
    # reading file
    ds = nc.Dataset(file)
    
    # getting arrays for raw data arrays and quality arrays for each parameter
    mask_arT = ds['TEMP'][:]
    mask_arP = ds['PRES'][:]
    mask_arS = ds['PSAL'][:]
    qc_arT = ds['TEMP_QC'][:]
    qc_arP = ds['PRES_QC'][:]
    qc_arS = ds['PSAL_QC'][:]
    
    T, P, S = quality_data() # getting only good quality data
    corrected_S = S_new(S, T, P)
    # saving T, P, S dictionaries
    saved_dict[float_id]['PRES'] = P
    saved_dict[float_id][ 'PSAL_CORRECTED' ] = corrected_S
    saved_dict[float_id][ 'TEMP' ] = T
    
    # getting reference time from the file
    reference = reference_utc( ds['REFERENCE_DATE_TIME'][:].data )
    # getting julian time for each profile and converting to utc and saving them as a dictionary
    juld = ds['JULD_LOCATION' ][:].data
    
    time_dict = {}
    julian_dict = {}
    for i in range(len(juld)):
        time_dict[i] = julian_time_to_utc(reference, juld[i])
        julian_dict[i] = juld[i]
        
    saved_dict[float_id][ 'TIME_GREGORIAN'  ] = time_dict
    saved_dict[float_id][ 'JULIAN_TIME' ] = julian_dict
    
    # getting latitudes and longitudes of profiles and saving them as dictionary
    lat_dict = {}
    long_dict = {}
    lat = ds['LATITUDE'][:].data
    long = ds['LONGITUDE'][:].data
    
    for i in range(len(lat)):
        lat_dict[i] = lat[i]
        long_dict[i] = long[i]
    
    saved_dict[float_id][ 'LATITUDE' ] = lat_dict
    saved_dict[float_id][ 'LONGITUDE' ] = long_dict
    
    


# In[ ]:


saved_file = 'float_raw_data.txt'
with open(saved_file, 'wb') as f:
    pickle.dump(saved_dict, f, pickle.HIGHEST_PROTOCOL)    


# In[ ]:


with open(saved_file, 'rb') as f:
    dict_saved = pickle.load(f)


# In[ ]:


for ar in dict_saved['3902327']["PRES"]:
    print(dict_saved['3902327']["PRES"][ar][-1])


# In[2]:


def quality_data( t, p, s ):
    
    # getting masked_arrays of temperature, pressure, and salinity, and also the qc (quality control) arrays
    
    global qc_arT
    global qc_arP
    global qc_arS
    
    
    # indices of qc arrays where qc value 2
    
    valid_id01 = np.where( qc_arT.data == 2 )
   
    valid_id1 = [ (valid_id01[0][i], valid_id01[1][i]) for i in range(len(valid_id01[0]))  ]
    
    valid_id02 = np.where(qc_arP.data == 2)
    valid_id2 = [ (valid_id02[0][i], valid_id02[1][i]) for i in range(len(valid_id02[0]))  ]
    
    valid_id03 = np.where(qc_arS.data == 2)
    valid_id3 = [ (valid_id03[0][i], valid_id03[1][i]) for i in range(len(valid_id03[0]))  ]
   
    
    valid_id_set = set.intersection(set(valid_id1), set(valid_id2), set(valid_id3)) # A set of tuples (these tuples will contain just valid integer indices) 


    # initiating a dictionary to catch valid values for T, P, and S
    dictT = {} 
    dictP = {}
    dictS = {} 
    
    for i in range(len(qc_arT)): # len(qc_array) = number of cycles
        dictT[ i ] = [] # iniitating an empty list for cycle i + 1; i+1 to ensure that first cycle no = 1  
        dictP[ i ] = []
        dictS[ i ] = []
   
        for j in range(len(qc_arT[i])): # len(mask_array[i]) = number of observations in a cycle
            if (i,j) in valid_id_set:
                dictT[i].append( t[i][j] )
                dictP[i].append( p[i][j] )
                dictS[i].append( s[i][j] )
        if len(dictP[i]) == 0:
            del dictP[i]
            del dictS[i]
            del dictT[i]
    
                
    return dictT, dictP, dictS


# In[3]:


def julian_time_to_utc(julian_year_start_utc, julian_time):
    # Extract year, month, day, hour, minute, second from Julian year start
    year = int(julian_year_start_utc[:4])
    month = int(julian_year_start_utc[5:7])
    day = int(julian_year_start_utc[8:10])
    hour = int(julian_year_start_utc[11:13])
    minute = int(julian_year_start_utc[14:16])
    second = int(julian_year_start_utc[17:19])

    # Convert Julian year start UTC to a datetime object
    dt_start_utc = datetime.datetime(year, month, day, hour, minute, second)

    # Calculate the datetime for the given Julian time
    dt = dt_start_utc + datetime.timedelta(days=julian_time)

    return dt


# In[4]:


file_list = ['1992_P16AandP17A.nc', '1993_P17E:P19S.nc', '1992_S04P.nc', '1994_P18.nc', '2007_P18.nc', '2011_S04P.nc', '2016_P18.nc', '2017_P17E.nc', '2018_S04P.nc']


# In[ ]:


# saving good quality data for each float

saved_dict = {}
for file in file_list:
    
    float_id = file[0:-3]
    saved_dict[float_id] = {}
    # reading file
    ds = nc.Dataset(file)
    
    # getting arrays for raw data arrays and quality arrays for each parameter
    t = ds['ctd_temperature'][:]
    p = ds['pressure'][:]
    s = ds['ctd_salinity'][:]
    qc_arT = ds['ctd_temperature_qc'][:]
    qc_arP = ds['pressure_qc'][:]
    qc_arS = ds['ctd_salinity_qc'][:]
    
    T, P, S = quality_data(t, p, s) # getting only good quality data
    
    # saving T, P, S dictionaries
    saved_dict[float_id]['PRES'] = P
    saved_dict[float_id][ 'PSAL_CORRECTED' ] = S
    saved_dict[float_id][ 'TEMP' ] = T
    
    # getting reference time from the file
    reference = '1950-01-01T00:00:00+00:00'
    # getting julian time for each profile and converting to utc and saving them as a dictionary
    juld = ds['time' ][:].data
    
    time_dict = {}
    julian_dict = {}
    for i in P:
        time_dict[i] = julian_time_to_utc(reference, juld[i])
        julian_dict[i] = juld[i]

    saved_dict[float_id][ 'JULIAN_TIME'  ] = julian_dict
    saved_dict[float_id][ 'TIME_GREGORIAN'  ] = time_dict
    
    # getting latitudes and longitudes of profiles and saving them as dictionary
    lat_dict = {}
    long_dict = {}
    lat = ds['latitude'][:].data
    long = ds['longitude'][:].data
    
    for i in P:
        lat_dict[i] = lat[i]
        long_dict[i] = long[i]
    
    saved_dict[float_id][ 'LATITUDE' ] = lat_dict
    saved_dict[float_id][ 'LONGITUDE' ] = long_dict
    
    btm_depth_dict = {}
    btm_depths = ds['btm_depth'][:].data
    
    for i in P:
        btm_depth_dict[i] = btm_depths[i]
        
    saved_dict[float_id][ 'btm_depth' ] = btm_depth_dict


# In[ ]:


saved_file = 'ship_raw_data.txt'
with open(saved_file, 'wb') as f:
    pickle.dump(saved_dict, f, pickle.HIGHEST_PROTOCOL)  


# In[ ]:


saved_file = 'float_raw_data.txt'
with open(saved_file, 'rb') as f:
    dict_saved1 = pickle.load(f)


# In[ ]:


dict_saved1.keys()


# In[ ]:


dict_saved1['5906831']['TIME_GREGORIAN']


# In[ ]:


dict_saved1['1992_S04P']['JULIAN_TIME'][20]


# In[ ]:


len(dict_saved1['1992_S04P']['PSAL_CORRECTED'])


# In[ ]:




