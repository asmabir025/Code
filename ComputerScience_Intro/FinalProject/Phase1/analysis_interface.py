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
import random


# In[ ]:





# In[2]:


MAIN_OPTIONS = [  'Four Characteristic Plots of one cycle/multiple cycles of one or more specific float', \
                'Four Characteristic Plots of all floats', \
               'Four Characteristic Plots in specific geographic space',\
               'Four Characteristic Plots in specific time range ']
FLOAT_IDS = [ '3902325', '3902326', '3902327','3902328', '5906830', '5906831', '5906832', '5906833'] 


# In[3]:


def main_menu():
    ''' Prints options for the user 
    '''
    border = 60 * '*'
    print(border)
    print("Options")

    for index in range(len(MAIN_OPTIONS)):
        print(index + 1, MAIN_OPTIONS[index])

    print(border)


# In[4]:


def select_float_id():
    border = 60 * '*'
    print(border)
    print("Float IDs:")
    
    for index in range(len(FLOAT_IDS)):
        print(index, FLOAT_IDS[index])
        
    input_str = input('Select one or more (if more than one, for example, 0 1 2):  ').split()
    
    float_id_list = [int(string) for string in input_str]
        
    print(border)
    
    return float_id_list


# In[5]:


def select_task():
    main_menu()

    selection = input("Enter a number from the menu: ")
    if selection.isnumeric() == True:
        selection = int(selection)
    if 0 < selection < 5:
        print(selection)
        return selection

    print("Invalid selection. You must enter a number from 1 to 4.")
    select_option()
    


# In[6]:


def select_cycle():
    global float_id_list
    global dict_saved
    
    print( "If you would like to plot n sequential cycles, type 'START n'.")
    print( "If n random cycles, type 'n'." )
    print(f'START can be 0. n must be less than {n}')
    
    cycle_dict = {}
    for float_id in float_id_list:

        n = len(dict_saved[float_id]['TIME'].keys())
        input_str = input(f'Number of cycles you would like to plot for float ID {float_id}:').split()

        if len(input_str)>1 :
            first = int(input_str[0])
            cycle_dict[float_id] = [ i for i in range(first, first+int(input_str[1])) ]

        elif len(input_str) == 1:
            all_cycles = [ i for i in range(n)]
            cycle_dict[float_id] = random.sample(all_cycles, int(input_str[0]))
    
    return cycle_dict


# In[8]:


def cycle_float_plot():
    global float_id_list
    global cycle_dict
    global dict_saved
    ct = []
    pres = []
    sa = []
    sigma4 = []
    
    for float_id in float_id_list:
        cycle_list = cycle_dict[float_id]
        for cycle in cycle_list:
            ct += dict_saved[float_id]['C_T'][cycle]
            pres += dict_saved[float_id]['PRES'][cycle]
            sa += dict_saved[float_id]['SA'][cycle]
            sigma4 += dict_saved[float_id]['Sigma4'][cycle]
        
    fig, axs = plt.subplots(2, 2)
    
    axs[0,0].plot(ct, pres) # plotting just one cycle
    #axs[0,0].title(f'Deep Argo Float {file[:7]}', fontdict={'weight': 'bold'})
    axs[0,0].ylim(0, max(pres)) # using ylim to avoid unnecessary extra axis lines
    axs[0,0].set_xlabel(r'$C_T$ ($^\circ$C)', fontsize=20)
    axs[0,0].set_ylabel('P (dbar)', fontsize=20)
    axs[0,0].gca().invert_yaxis() # trend of hydrographic profile: lowest pressure (sea level) at top
    axs[0,0].gca().spines['bottom'].set_position(('data', 0)) # to move the x-axis on top
    axs[0,0].gca().xaxis.tick_top() # to move x-ticks on top
    axs[0,0].gca().xaxis.set_label_position('top') # to move x-label on top
    axs[0,0].gca().spines['top'].set_visible(False) # to avoid unnecessary line on x-axis
    
    axs[0,1].plot(sa, pres) # plotting just one cycle
    #axs[0,0].title(f'Deep Argo Float {file[:7]}', fontdict={'weight': 'bold'})
    axs[0,1].ylim(0, max(pres)) # using ylim to avoid unnecessary extra axis lines
    axs[0,1].set_xlabel('SA (g/kg)', fontsize=20)
    axs[0,1].set_ylabel('P (dbar)', fontsize=20)
    axs[0,1].gca().invert_yaxis() # trend of hydrographic profile: lowest pressure (sea level) at top
    axs[0,1].gca().spines['bottom'].set_position(('data', 0)) # to move the x-axis on top
    axs[0,1].gca().xaxis.tick_top() # to move x-ticks on top
    axs[0,1].gca().xaxis.set_label_position('top') # to move x-label on top
    axs[0,1].gca().spines['top'].set_visible(False) # to avoid unnecessary line on x-axis
    
    axs[1,0].plot(sigma4, pres) # plotting just one cycle
    #axs[1,0].title(f'Deep Argo Float {file[:7]}', fontdict={'weight': 'bold'})
    axs[1,0].ylim(0, max(pres)) # using ylim to avoid unnecessary extra axis lines
    axs[1,0].set_xlabel(r'$\sigma_4$ ($kgm^{-3}$)', fontsize=20)
    axs[1,0].set_ylabel('P (dbar)', fontsize=20)
    axs[1,0].gca().invert_yaxis() # trend of hydrographic profile: lowest pressure (sea level) at top
    axs[1,0].gca().spines['bottom'].set_position(('data', 0)) # to move the x-axis on top
    axs[1,0].gca().xaxis.tick_top() # to move x-ticks on top
    axs[1,0].gca().xaxis.set_label_position('top') # to move x-label on top
    axs[1,0].gca().spines['top'].set_visible(False) # to avoid unnecessary line on x-axis
    
    S, T = np.meshgrid(sa, ct)
    potential_den = gsw.sigma4(S, T)
    contour = axs[1,1].contour(S, T, potential_den, colors='k')
    axs[1,1].plot(sa, ct, color='r')
    # label_positions = [(sa_val, ct_val) for sa_val, ct_val in zip(sa[cycle], ct[cycle])]
    # ax.clabel(contour, inline=True, fontsize=8, manual=label_positions)
    axs[1,1].clabel(contour, inline=True, inline_spacing=3, rightside_up=True, colors='k', fontsize=8)
    #plt.title(f'Deep Argo Float {float_id}', fontdict={'weight': 'bold'})
    #ax.set_ylim(min(ct[cycle]), max(ct[cycle]))
    axs[1,1].set_xlabel('SA (g/kg)', fontsize=20)
    axs[1,1].set_ylabel('$C_T$ ($^\circ$C)', fontsize=20)
#     plt.rc('figure', figsize=(6, 6))
#     plt.legend(loc=3)
#     plt.savefig('PressurevsSal.png')
#     plt.show()


# In[7]:


def main():
    saved_file = 'float_calc_quantities.txt'
    with open(saved_file, 'rb') as f:
        dict_saved = pickle.load(f)

    selection = select_task()
    if selection == 1:
        float_id_list = select_float_id()
        cycle_dict = select_cycle()
        cycle_float_plot()


# In[ ]:




