#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# change the following to %matplotlib notebook for interactive plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')


# In[2]:


import numba
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import pickle


# In[3]:


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})


# In[39]:


size_list = ['.5', '1', '2']
frame_rate_list_DIA = [5, 4.74, 5.01] #frame rate for different video files in DIA
frame_rate_list_EPI = [3.8, 3.85, 3.24]
max_displacement = [10,18,5] #maximum displacement between two frames for each size
color_list = ['b', 'c', 'r']
unit = r'$\mu$m'
slope_list_dia = [] #initiating a list for catching slopes for three sizes
uncern_dia = [] #initiating a list for catching uncertainties for three sizes


#This section of code for DIA files
fig, axs = plt.subplots(2, 2)
for i in range(len(size_list)):
   
    #if i == 1:
    file_name = '0'+ size_list[i] + "m02min_DIA.txt" 
    file = open(file_name, "rb")
    f = pickle.load(file)


    t = tp.link(f, max_displacement[i], memory=4)
    t1 = tp.filter_stubs(t, 10)
    d = tp.compute_drift(t1)
    tm = tp.subtract_drift(t1.copy(), d)
    t1 =tm
    
    
    em = tp.emsd(t1, 0.16, frame_rate_list_DIA[i]) 
    msd_data = em.values
    times_data = np.array(em.index)
    
    
    
    
    axs[0, 0].scatter(times_data, msd_data, marker = '.', c=color_list[i])
    
    temp_slope_dia = []
    uncern_slope_dia = []
    for j in range(0, 50, 50):
        

        msd_data1 = msd_data[j:j+50]
        times_data1 = times_data[j:j+50]
        #print(msd_data1)

        slope, cov = np.polyfit(times_data1, msd_data1, 1, cov=True)
        slope = np.poly1d(slope)
        temp_slope_dia+=[slope[1]]
        uncern_slope_dia.append(np.sqrt(cov[1][1]))

        line = [slope[1] * y1 + slope[0] for y1 in times_data1]
        
#         if i == 0:
#             axs[0, 1].plot(times_data1, line, color = 'white')
#         elif i == 1:
        
#             axs[0, 1].plot(times_data1, line, color = 'black')
            
#         else:
#             axs[0,1].plot(times_data1, line, color = 'yellow')
            
        axs[0, 1].scatter(times_data1, msd_data1, color=color_list[i])
       
    weights_d = 1.0/np.array(uncern_slope_dia)**2
    WeightedAvg_d = np.dot(np.array(temp_slope_dia),weights_d)/np.sum(weights_d)
    α_d = np.sqrt(1.0/np.sum(weights_d))
    
    slope_list_dia.append(WeightedAvg_d)
    uncern_dia.append(α_d)

    
   
#This section of code for EPI files    
slope_list_epi = []
uncern_epi = []

for i in range(len(size_list)):
    #if i == 1:
    file_name = '0' + size_list[i] + "m02min_EPI.txt" 
    file = open(file_name, "rb")
    f = pickle.load(file)


    t = tp.link(f, max_displacement[i], memory=4)
    t1 = tp.filter_stubs(t, 10)
    
   
    d = tp.compute_drift(t1)
    tm = tp.subtract_drift(t1.copy(), d)
    t1 =tm
    
    em = tp.emsd(t1, 0.16, frame_rate_list_EPI[i]) 
    msd_data = em.values
    times_data = np.array(em.index)
    
    axs[1, 0].scatter(times_data, msd_data, marker = '.', c=color_list[i], label=f'0{size_list[i]}{unit}')
    
    axs[1,0].legend()
    
    temp_slope_epi = []
    uncern_slope_epi = []
    for j in range(0, 50, 50):
        
        if i == 0 and j <= 25:
            msd_data1 = msd_data[j:j+25]
            times_data1 = times_data[j:j+25]
            
        elif i > 0 :
            msd_data1 = msd_data[j:j+50]
            times_data1 = times_data[j:j+50]

        slope, cov = np.polyfit(times_data1, msd_data1, 1, cov=True)
        slope = np.poly1d(slope)

        line = [slope[1] * y1 + slope[0] for y1 in times_data1]
        temp_slope_epi+=[slope[1]]
        uncern_slope_epi.append(np.sqrt(cov[1][1]))
        
        axs[1, 1].scatter(times_data1, msd_data1, color=color_list[i])
#         if i == 0:
#             axs[1, 1].plot(times_data1, line, color = 'white')
#         elif i == 1:
        
#             axs[1, 1].plot(times_data1, line, color = 'black')
            
#         else:
#             axs[1,1].plot(times_data1, line, color = 'yellow')
    weights_e = 1.0/np.array(uncern_slope_epi)**2
    WeightedAvg_e = np.dot(np.array(temp_slope_epi),weights_e)/np.sum(weights_e)
    α_e = np.sqrt(1.0/np.sum(weights_e))
    
    slope_list_epi.append(WeightedAvg_e)
    uncern_epi.append(α_e)

    
  
axs[0, 0].set_title('Microscopy Method')
axs[0, 1].set_title('Linear Regime')

axs[0,0].text(5, 30,'DIA', fontsize='large') 
axs[1,0].text(17, 5,'EPI', fontsize='large')

rect1 = patches.Rectangle((4.5, 29.5), 4, 5, edgecolor='black', facecolor='none')
rect2 = patches.Rectangle((16, 4.5), 6, 6.5, edgecolor='black', facecolor='none')
axs[0,0].add_patch(rect1)
axs[1,0].add_patch(rect2)

axs[1, 0].set_xlabel(r'lag time $t(s)$')
axs[0, 0].set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.gcf().set_size_inches(8,7)
plt.savefig('Method_size.png')


# In[40]:


slope_list_dia, slope_list_epi


# In[41]:


R = 8.31
eta = 9.544e-4
T = 295
Na = 6.023e23
size_list = [0.474,1.354,1.840]
D_list = [R*T/(3*np.pi*Na*(n/2)*1e-6*eta) for n in size_list]


# In[42]:


D_list


# In[43]:


plt.figure()
size_list = [0.474,1.354,1.840]
diff_dia = [1e-12*n/2 for n in slope_list_dia]
diff_epi = [1e-12*n/2 for n in slope_list_epi]
uncern_dia_diff = [1e-12*n/2 for n in uncern_dia]
uncern_epi_diff = [1e-12*n/2 for n in uncern_epi]
plt.errorbar(size_list, diff_dia, uncern_dia_diff, fmt= "xb", label = 'DIA')
plt.errorbar(size_list, diff_epi, uncern_epi_diff, fmt= "Dr", label = 'EPI')
plt.scatter(size_list, D_list, color = 'cyan', label = 'Predicted')
plt.xlabel(r'Size 2r ($\mu$m)')
plt.ylabel(r'Diffusion Coefficient D ($m^2s^{-1}$)')
plt.gcf().set_size_inches(6.5,4.5)
plt.legend(loc = 'upper center')
plt.savefig('slope_plot.pdf')


# In[44]:


#Calulating Avogadro Number

R = 8.31

a_list = [(0.474/2)*1e-6, (1.354/2) * 1e-6, (1.840/2) * 1e-6] # 2 um

eta = 9.544e-4
T = 295
uncert_T = 2

A_dia = [] # Avogadro Number list for DIA files
A_epi = [] # Avogadro Number list for EPI files
dia_er = []
epi_er = []
for i in range(len(slope_list_dia)):
    A1 = slope_list_dia[i]*10**(-12)
    A2 = slope_list_epi[i]*10**(-12)
    A_dia.append( 4*R*T / (6*np.pi*a_list[i]*eta*A1))
    A_epi.append(  4*R*T / (6*np.pi*a_list[i]*eta*A2))
    dia_er.append(np.sqrt((4*R*T / (6*np.pi*a_list[i]*eta*A1) - 4*R*T / (6*np.pi*a_list[i]*eta*(A1+uncern_dia[i]*1e-12)))**2 + (4*R*T / (6*np.pi*a_list[i]*eta*A1) - 4*R*(T-uncert_T) / (6*np.pi*a_list[i]*eta*A1))**2)) 
    epi_er.append(np.sqrt((4*R*T / (6*np.pi*a_list[i]*eta*A2) - 4*R*T / (6*np.pi*a_list[i]*eta*(A2+uncern_epi[i]*1e-12)))**2 + (4*R*T / (6*np.pi*a_list[i]*eta*A2) - 4*R*(T-uncert_T) / (6*np.pi*a_list[i]*eta*A2))**2)) 
   


# In[38]:


plt.figure()
plt.errorbar(size_list, A_dia, dia_er, fmt= "xb", label = 'DIA')
plt.errorbar(size_list, A_epi, epi_er, fmt= "Dr", label = 'EPI')
plt.axhline(6.023e23, color = 'cyan', label='Predicted')
plt.ylabel(r"$N_A$")
plt.xlabel(r'2r ($\mu$m)')
plt.legend()
plt.gcf().set_size_inches(6.5,4.5)
plt.savefig('N_A_plot.png')


# In[ ]:




