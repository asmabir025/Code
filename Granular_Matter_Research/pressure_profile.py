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
import statistics as stats
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
from scipy import optimize
import urllib
import math


# In[2]:


mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (6, 5)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# In[4]:


# you can modify path_name function. The function currently returns directory address
#of dump files on my local computer. 

def path_name(pin_num):
    '''
    returns full path to dump files.
    Parameters:
        - cwd (current working directory); str 
        - gammadot; str
        - pin_num; int
        - phi; float
        - filename; str
    returns path; str
            
    '''
    cwd = os.getcwd()                          #getting current directory
    #dir_name =  str(gammadot) + '/'            #example: 1e-06/
    #dir_name += 'Pins' + str(pin_num) + '/'    #example: 1e-06/Pins9
    dir_name = 'Pins' + str(pin_num) + '/'    #example: 1e-06/Pins9
    dir_name += 'Dump_Files/'                  #example: 1e-06/Pins9/Phi084500/Dump_Files/
            
    path = os.path.join(cwd, dir_name)
    #example: [current directory]/1e-06/Pins9/Phi084500/Dump_Files/
    
    return path


# In[5]:


def read_data_MD(pathname, MD):
    '''
    reads the id, type, xu, yu columns of the dump file
    parameters:
    pathname; str; the full path to dump_files directory
    MD; str; particular MD step number
    '''
    global file_str
    file = pathname + f'{file_str}{MD}.data'
    # skipping 1st 9 lines
    # 1st column: id; 2nd: type; 3rd column: xu; 4th column: yu
    typearray, yarray, Pxarray, Pyarray = np.loadtxt(file, skiprows=9, usecols = (1,3,11,12), dtype='float', unpack=True)
    return typearray, yarray, Pxarray, Pyarray


# In[6]:


def getting_wall(pathname):

    wall_file = pathname + 'wallpartidtype'
    
    wallids = np.loadtxt(wall_file, skiprows=9, usecols=(0), dtype='int', unpack=True)
    
    return wallids


# In[7]:


def localP(typearray, y_array, Pxarray, Pyarray, wallids):
    
    P = -0.5*(Pxarray + Pyarray)
    
    wallid_indices = [ids-1 for ids in wallids]
    pin_indices = [i for i in range(len(typearray)) if typearray[i]==3]
    ids_to_remove = wallid_indices + pin_indices
    
    P_midmove = np.delete(P, ids_to_remove)
    y_midmove = np.delete(y_array, ids_to_remove)
        
    
    return P_midmove, y_midmove


# In[8]:


def layers_y_coordinates():

    global Na, Nb, phi, Ra, Rb 
    
    Lx = np.sqrt(np.pi*((Ra**2)*Na+(Rb**2)*Nb)/phi)
    
    bottom_wall_top = 3*Rb*2
    walltopystart = Lx + bottom_wall_top
    layer_thickness = Lx/32
    layer_area = layer_thickness*Lx
    
    layers_y = []
    for y in np.arange(bottom_wall_top, walltopystart, layer_thickness):
        layers_y.append(y + layer_thickness/2)
        
    return bottom_wall_top, layer_thickness, layers_y, layer_area


# In[9]:


def pressure_profile(layers_y, y_midmove, P_midmove):
    global bottom_wall_top, layer_thickness, layer_area
    
    pressure_dict = {}
    delta_pres_dict = {}
    for y_coord in layers_y:
        pressure_dict[y_coord] = []
        
#     print(pressure_dict)
    for i in range(len(y_midmove)):
        layer_num = int((y_midmove[i] - bottom_wall_top) / layer_thickness)
        
        if 0 <= layer_num <= 31:
            pressure_dict[layers_y[layer_num]] += [P_midmove[i]]
            
    for keys in pressure_dict:
        
        delta_pres_dict[keys] = len(pressure_dict[keys])*stats.stdev(pressure_dict[keys])/(np.sqrt(len(pressure_dict[keys]))*layer_area)
        pressure_dict[keys] = np.sum(pressure_dict[keys])/layer_area
        
    return pressure_dict, delta_pres_dict


# In[10]:


def all_pres_prof(MDstep_list):
    
    global layers_y
    
    all_pres_dict = {}
    
    for MD_step in MDstep_list:
        
        typearray, yarray, Pxarray, Pyarray = read_data_MD(path, int(MD_step))
           
        P_midmove, y_midmove = localP(typearray, yarray, Pxarray, Pyarray, wallids)
    
        pressure_dict, delta_pres_dict = pressure_profile(layers_y, y_midmove, P_midmove)
        
        all_pres_dict[MD_step] = (pressure_dict, delta_pres_dict)
        
    return all_pres_dict


# In[11]:


def plot_press_prof(pressure_dict, delta_pres_dict):
    
    global MD_step
    
    y_vals = list(pressure_dict.keys())
    pres_vals = list(pressure_dict.values())
    unc_pres = list(delta_pres_dict.values())
    
    plt.figure()
    
    plt.errorbar(y_vals, pres_vals, unc_pres, fmt='o')
    plt.xlabel('y', fontsize=20)
    plt.ylabel('P', fontsize=20)
    plt.text(0.5, 1.01, f'MDstep = {MD_step}', transform=plt.gca().transAxes)
    plt.savefig('test.png')


# In[12]:


def range_pres(all_pres_dict):
    min_list = []
    max_list = []
    for MD_step in all_pres_dict:
        pres_vals = list(all_pres_dict[MD_step][0].values())
        min_list.append(min(pres_vals))# appending minimum vx for the layer
        max_list.append(max(pres_vals))# appending maximum vx for the layer 
        
    min_quantity = min(min_list) # minimum vx 
    max_quantity = max(max_list) # maximum vx
    
    return min_quantity - min_quantity*0.25, max_quantity + max_quantity*0.25


# In[13]:


def animate(n):
    '''Creates n th frame of the animation
    '''

    global all_pres_dict
    global min_pres, max_pres
    global MDstep_list
    
    # Getting the scale of vx axis 
    
    pressure_dict = all_pres_dict[MDstep_list[n]][0]
    delta_pres_dict = all_pres_dict[MDstep_list[n]][1]
    y_vals = list(pressure_dict.keys())
    pres_vals = list(pressure_dict.values())
    unc_pres = list(delta_pres_dict.values())
    
    plt.cla()
    
    plt.errorbar(y_vals, pres_vals, unc_pres, fmt='o', markersize=5)
    plt.xlabel('y', fontsize=20)
    plt.ylabel('P', fontsize=20)
    plt.ylim(min_pres, max_pres)
    plt.text(0.5, 1.01, f'MDstep = {MDstep_list[n]}', transform=plt.gca().transAxes)


# In[14]:


def animate_profile(): 
    '''Parameters:
    - num_MDsteps; int; total MD steps
    - quantity_str; string; for example, 'vx'
    Creates animation of profile and saves it
    '''
    # styles
    plt.rcParams["animation.html"] = "jshtml"
    #This dictionary controls the default behavior for all your matplotlib plots 
    #such as figure size, font size, and how your animations should display when you call them.
    plt.rcParams['figure.dpi'] = 150  #to increase the image quality
    plt.ioff() #werid things that makes it works #prevents newly created figures to be shown immediately 

    global pin
    global region
    global file_str
    global MDstep_list
    
    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(MDstep_list), interval=1000)
    #frame=Number of profiles/mdsteps
    # interval= how much wait time before showing next frame (in ms)
    # animate is the function defined later on
    
    name = f"pins{pin}_pres_prof_{region}.MP4" 
    #example: [current directory]/1e-06/Pins9/Phi084500/gammadot1e-06_pins9_vxprof_inf.MP4
    
    writervideo = matplotlib.animation.FFMpegWriter(fps=1) # fps = frame per second
    ani.save(name, writer=writervideo) # saving the animation
    plt.show()


# In[16]:


pin_list = [225]
region = 'bounce'
file_str = f'confdumpall{region}MD' # 'confdumpallelastic' or 'confdumpalltransient' or 'confdumpallbounce'
MD_initial = 0
MD_final = 5000000
delta_MD = 500000
MDstep_list = [i for i in range(MD_initial, MD_final+delta_MD, delta_MD)]

Na = 2048
Nb = Na
Ra = 1
Rb = 1.4*Ra
phi = 0.845
bottom_wall_top, layer_thickness, layers_y, layer_area = layers_y_coordinates()

for pin in pin_list:
    path = path_name(pin)
    wallids = getting_wall(path)
    
    
    all_pres_dict = all_pres_prof(MDstep_list)
    min_pres, max_pres = range_pres(all_pres_dict)
    
    animate_profile()


# In[ ]:





# In[ ]:




