#!/usr/bin/env python
# coding: utf-8

# For getting a single particle's trajectory and properties and plotting them, go to the block next to
# # Extracting particle info from dump files
# 
# AND
# 
# # Plotting particle's trajectory and properties
# 
# # -------------------------------------------------------

# In[15]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
import pickle
#%matplotlib notebook


# In[4]:


mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (6, 6)) # Reduces overall size of figures
plt.rc('axes', labelsize=32, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# In[20]:


def path_name(gammadot, pin_num, phi):
    '''
    creates directory names that are in my stampede2 directory and contain the data for aps meeting 
    Parameters:
        - cwd (current working directory); str 
        - gammadot; float
        - pin_num; int
        - phi; float
        - filename; str
    returns path; str
            
    '''
    cwd = os.getcwd()
    dir_name =  str(gammadot) + '/'                                             #example: 1e-06/
    dir_name += 'Pins' + str(pin_num) + '/'                                #example: 1e-06/Pins0
    dir_name += 'Phi0' + str(int(phi*1e5)) + '/'                        #example: 1e-06/Pins0/Phi083500
    dir_name += 'Dump_Files/'                                   #example: 1e-06/Pins0/Phi083500/Dump_Files
                    
    
    path = os.path.join(cwd, dir_name)
    
    return path


# In[12]:


def read_d2min(gammadot):
    '''reads d2min values from a simulation output of a particular shear rate
    Parameter:
    gammadot; value of gammadot, for example: 1e-06; float
    return d2min_dict; dictionary
           key(s): shear rate (int)
           value(s): a nested dictionary
                   Key(s): Pin number (int)
                   value(s): a nested dictionary 
                            key(s): Phi value (int)
                            value(s): a nested dictionary
                                     keys: MDstep value (int)
                                     values: a nested dictionary 
                                            keys: particle IDs
                                            values: d2min (float)
                   
    '''
    filename = 'd2min_' + str(gammadot) + '.txt'
    
    with open(filename, "rb") as fp:
    
        d2min_dict = pickle.load(fp)
    return d2min_dict


# In[13]:


def particle_info(MD_step, pathname, id_no):
    '''returns total particle number (Ntot) and half-length of box in x-direction from the reference file
    Parameters:
    MD_step; int
    pathname; path to the folder where the dump file is; str
    id_no; particle ID; int
    
    returns
    x_pos; x-coordinate of the particle; float
    y_pos; y-coordinate of the particle; float
    sigmaxy; shear stress of the particle; float
    type_i; particle type; int
    '''
    global file_str
    ref_file = pathname + f'{file_str}{MD_step}.data' 
    open_file = open(ref_file, 'r')
    countline = 0

    # Reading in reference file to determine Ntot, Lx for the configuration
    
    for line in open_file.readlines():
        countline +=1
        
        if countline == 9 + id_no: # Assuming that dump file lines are sorted according to particle ID
            fields = line.split()
            type_i = int(fields[1])
            x_pos = float(fields[8])
            y_pos = float(fields[9])
            sigmaxy = float(fields[10])
            
            break

    open_file.close()
    
    return x_pos, y_pos, sigmaxy, type_i


# # Extracting particle info from dump files

# In[22]:


''' Only next few lines you might need to change 
'''    
gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [9]
phi_list = [0.845]
total_dump_files = 101 # Number of dump files we are analyzing 
inf_strain_max = 0.1  # total amount of strain occured in these dump files
id_no = 788 # particle ID; whatever your choice is
file_str = 'confdumpallinfMD'  # could be 'confdumpallelastic' or anything you name for the dump files

for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        
        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j])) # shear rate
        #print(gammadot)
        
        strain_per_line = gammadot*0.01 # strain in one MD step
        #print(strain_per_line)

        total_MDsteps = inf_strain_max/strain_per_line # total MD steps ran in between the first and last dump files
        #print(total_MDsteps)

        dump_frequency =  total_MDsteps/(total_dump_files-1) # how many MD steps are these dump files apart
        #print(dump_frequency)

        MDstep_list = [int(i*dump_frequency) for i in range(total_dump_files)] # list of MD steps of dump files
        
        d2min_dict = read_d2min(gammadot) # reading d2min values 
        
        
        for k in range(len(pin_list)):
            pin_num = pin_list[k]
    
            for l in range(len(phi_list)):

                phi_val = phi_list[l]
                # first getting the full path to the file
                path = path_name(gammadot, pin_num, phi_val)
                
                x_list = [] # initializing a list to catch a particle's x-positions over time 
                y_list = [] # initializing a list to catch a particle's y-positions over time
                sigmaxy_list = [] # initializing a list to catch a particle's shear stress over time
                t_list = [] # initializing a list to catch time values
                d2min_list = [] # initializing a list to catch a particle's d2min values over time

                for m in range(len(MDstep_list)-1):

                    # getting particle information at that MD step
                    x,y,sigmaxy, type_i = particle_info(MDstep_list[m], path, id_no)
                    d2min = d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m]][id_no]
                    
                    # updating the lists of time, and informations of particle
                    t_list.append(MDstep_list[m]*gammadot*0.01)
                    x_list.append(x)
                    y_list.append(y)
                    sigmaxy_list.append(sigmaxy)
                    d2min_list.append(d2min)


# # plotting particle's trajectory and properties 

# In[24]:


# plotting x vs t 
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
plt.scatter(t_list, x_list)
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
#plt.xlim(0, 0.10) # you might need to change 
plt.savefig('x.png')


# In[25]:


# plotting y vs t 
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
plt.scatter(t_list, y_list)
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
#plt.xlim(0,0.1) # you might need to change 
plt.savefig('y.png')


# In[26]:


# plotting shear stress vs t 
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
plt.plot(t_list, sigmaxy_list, color = 'blue')
plt.xlabel(r'$t$')
plt.ylabel(r'$\sigma_{xy}$')
#plt.xlim(0,0.1) # you might need to change 
plt.savefig('sigmaxy.png')


# In[27]:


# plotting d2min vs t 
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
plt.plot(t_list, d2min_list, color = 'red')
plt.xlabel(r'$t$')
plt.ylabel(r'$D^2_{min}$')
#plt.xlim(0,0.1) # you might need to change 
plt.savefig('d2min.png')


# In[ ]:




