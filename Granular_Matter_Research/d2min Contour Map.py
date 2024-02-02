#!/usr/bin/env python
# coding: utf-8

# For creating animation of contour plots of d2min, go to the block next to
# # Creating animation of contour plots
# after running each code block before that.
# 
# For creating a d2min contour plot at a specific MD step, go to the block next to
# 
# # Plotting contour map of d2min at a particular MD step
# 
# after running the two code blocks: importing modules and setting style
# 
# 
# For creating a d2min contour plot at a specific MD step (particularly to view particles with extremely high d2min), go to the block next to
# 
# # Plotting contour map of d2min (particularly to view particles with extremely high d2min)
# 
# after running the two code blocks: importing modules and setting style

# # importing modules

# In[ ]:


import pickle 
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
import statistics as stats
get_ipython().run_line_magic('matplotlib', 'notebook')


# # setting style

# In[ ]:


mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (7, 7)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# In[ ]:


def animate(n):
    '''creates a single snapshot of the contour plot
    Parameter:
    n; integer; Dump File number (an indicator of MD step number)
    '''
    
    plt.cla()#clear the previous graph

  
    global distances_wall
    global distances_pin
    global MDstep_list
    global phi_val
    global pin_num
   
    ids = dist_dict[gammadot][pin_num][MDstep_list[n]]['id'] # getting the particle ID list
    # getting the minimum distance from pin (list)
    distances_pin = dist_dict[gammadot][pin_num][MDstep_list[n]]['min_pin_distance'] 
    # getting the minimum distance from wall (list)
    distances_wall = dist_dict[gammadot][pin_num][MDstep_list[n]]['min_wall_distance']

    # initiating lists to catch d2min values of particles, the corresponding minimum distances 
    # from pin and walls, respectively.
    
    sigma_d2mins = []  # d2min less or equal to mean + 1*(standard deviation)
    sigma_dist_wall = [] # corresponding minimum distance from wall list
    sigma_dist_pins = [] # corresponding minimum distance from pin list
    
    sigma2_d2mins = [] # d2min greater than mean + 1*(standard deviation) but less than mean + 2*(standard deviation)
    sigma2_dist_wall = [] # corresponding minimum distance from wall list
    sigma2_dist_pins = [] # corresponding minimum distance from pin list
    
    sigma3_d2mins = []  # d2min greater than mean + 2*(standard deviation)
    sigma3_dist_wall = [] # corresponding minimum distance from wall list
    sigma3_dist_pins = [] # corresponding minimum distance from pin list
    
    # getting the mean and standard deviation of d2min of all particles at certain MDstep n
    mean_d2min = stats.mean(list(d2min_dict[gammadot][pin_num][MDstep_list[n]].values())) 
    std_d2min = stats.stdev(list(d2min_dict[gammadot][pin_num][MDstep_list[n]].values())) 

    for i in range(len(ids)):
        
        d2min_value = d2min_dict[gammadot][pin_num][MDstep_list[n]][int(ids[i])] # getting d2min value of a particle
        # getting minimum distance from wall and pin, respectively, for the same particle
        dist_wall_val = dist_dict[gammadot][pin_num][MDstep_list[n]]['min_wall_distance'][i]
        dist_pins_val = dist_dict[gammadot][pin_num][MDstep_list[n]]['min_pin_distance'][i]

    
        if mean_d2min + std_d2min >= d2min_value: # d2min less or equal to mean + 1*(standard deviation)
            sigma_d2mins += [d2min_value]
            sigma_dist_wall += [dist_wall_val]
            sigma_dist_pins += [dist_pins_val]

        # d2min greater than mean + 1*(standard deviation) but less than mean + 2*(standard deviation)
        elif mean_d2min + 2*std_d2min >= d2min_value:
            sigma2_d2mins += [d2min_value]
            sigma2_dist_wall += [dist_wall_val]
            sigma2_dist_pins += [dist_pins_val]

        # d2min greater than mean + 2*(standard deviation)
        else:
            sigma3_d2mins += [d2min_value]
            sigma3_dist_wall += [dist_wall_val]
            sigma3_dist_pins += [dist_pins_val]
    
    # first plotting for lowest d2min values; these constitute the 'background' of the contour plot
    plt.scatter(sigma_dist_wall, sigma_dist_pins, color = 'lime', marker='.') 
    
    # then plotting for higher d2min values
    plt.scatter(sigma2_dist_wall, sigma2_dist_pins, color = 'blue', marker='.')
    plt.scatter(sigma3_dist_wall, sigma3_dist_pins, color = 'red', marker='.')
    
    plt.ylabel("Minimum Distance From Pins")
    plt.xlabel('Minimum Distance From Walls')
    plt.title(config_name)

#     plt.legend()
    


# In[ ]:


def animate_contour():
    '''creates the animation of all contour plots from all MDsteps
    '''
    #This dictionary controls the default behavior for all your matplotlib plots such as figure size,
    #font size, and how your animations should display when you call them.
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  #to increase the image quality
    plt.ioff() #werid things that makes it works #prevents newly created figures to be shown immediately 
    fig, ax = plt.subplots()
    
    global config_name
    global total_MDsteps
    
    # creating the animation by calling animate function
    ani = mpl.animation.FuncAnimation(fig, animate, frames=100, interval=1000) 
    #frame=block, interval= how much wait time before showing next frame
   
    writervideo = mpl.animation.FFMpegWriter(fps=1) # fps = frames per second
    ani.save(config_name, writer=writervideo) # saving the animation
    plt.show()


# # Creating animation of contour plots

# In[ ]:


'''Only next 5 lines you may need to change
'''

gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [9]
phi_list = [0.845]
total_dump_files = 101 # Number of dump files we are analyzing
inf_strain_max = 0.1 # total amount of strain occured in these dump files

for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        
        gammadot = gammadotbeforepoint_list[i]*10**(-gammadot_list[j])
        filename_dist = 'd2min_' + str(gammadot) + 'pinwall.txt'
        filename_d2min = 'd2min_' + str(gammadot) + '.txt'

        with open(filename_dist, "rb") as fp:
            dist_dict = pickle.load(fp)
        with open(filename_d2min, "rb") as fq:
            d2min_dict = pickle.load(fq)
        
        strain_per_line = gammadot*0.01 # strain in one MD step
        #print(strain_per_line) 

        total_MDsteps = inf_strain_max/strain_per_line # total MD steps ran in between the first and last dump files
        #print(total_MDsteps)

        dump_frequency =  total_MDsteps/(total_dump_files-1) # how many MD steps are these dump files apart
        #print(dump_frequency)

        MDstep_list = [int(i*dump_frequency) for i in range(total_dump_files)] # list of MD steps of dump files
        #print(MDstep_list)
        
        for k in range(len(pin_list)):
            pin_num = pin_list[k]
            config_name = str(gammadot) + '_' + str(pin_num) + '_d2min_contourmap.MP4'
            
            for l in range(len(phi_list)):
                phi_val = phi_list[l]
                animate_contour()                         


# # Plotting contour map of d2min at a particular MD step

# In[ ]:


n = 49 # index of MDstep list; determines dump file 
gammadot = 1e-6
pin_num = 9
phi_val = 0.845

# I have not documented this block of code because this is basically animate function 


strain_per_line = gammadot*0.01 # strain in one MD step
#print(strain_per_line) 

total_MDsteps = inf_strain_max/strain_per_line # total MD steps ran in between the first and last dump files
#print(total_MDsteps)

dump_frequency =  total_MDsteps/(total_dump_files-1) # how many MD steps are these dump files apart
#print(dump_frequency)

MDstep_list = [int(i*dump_frequency) for i in range(total_dump_files)] # list of MD steps of dump files
#print(MDstep_list)

ids = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['id']
distances_pin = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_pin_distance']
distances_wall = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_wall_distance']

    
sigma_d2mins = []
sigma_dist_wall = []
sigma_dist_pins = []
sigma2_d2mins = []
sigma2_dist_wall = []
sigma2_dist_pins = []
sigma3_d2mins = []
sigma3_dist_wall = []
sigma3_dist_pins = []

mean_d2min = stats.mean(list(d2min_dict[gammadot][pin_num][phi_val][MDstep_list[n]].values())) 
std_d2min = stats.stdev(list(d2min_dict[gammadot][pin_num][phi_val][MDstep_list[n]].values())) 

for i in range(len(ids)):
    d2min_value = d2min_dict[gammadot][pin_num][phi_val][MDstep_list[n]][int(ids[i])]
    dist_wall_val = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_wall_distance'][i]
    dist_pins_val = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_pin_distance'][i]


    if mean_d2min + std_d2min >= d2min_value: 
        sigma_d2mins += [d2min_value]
        sigma_dist_wall += [dist_wall_val]
        sigma_dist_pins += [dist_pins_val]

    elif mean_d2min + 2*std_d2min >= d2min_value:
        sigma2_d2mins += [d2min_value]
        sigma2_dist_wall += [dist_wall_val]
        sigma2_dist_pins += [dist_pins_val]

    else:
        sigma3_d2mins += [d2min_value]
        sigma3_dist_wall += [dist_wall_val]
        sigma3_dist_pins += [dist_pins_val]


fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0) 

plt.scatter(sigma_dist_wall, sigma_dist_pins, color = 'lime', marker='.', label = r'$ D^2_{min,i} \leq (<D^2_{min}> +1\sigma)$')
plt.scatter(sigma2_dist_wall, sigma2_dist_pins, color = 'blue', marker='.', label = r'$1\sigma < \Delta D^2_{min,i} \leq 2\sigma$')
plt.scatter(sigma3_dist_wall, sigma3_dist_pins, color = 'red', marker='x', label = r'$\Delta D^2_{min,i} > 2\sigma$')



plt.ylabel(r"Minimum Distance From Pins", fontsize=20)
plt.xlabel(r'Minimum Distance From Walls', fontsize=20)
# plt.title(config_name)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.legend(fontsize = 15, loc = 'upper right')
f_string = f'snapshot_{gammadot}_Pins{pin_num}_MD{n}contour.png'
plt.savefig(f_string)


# # Plotting contour map of d2min (particularly to view particles with extremely high d2min)

# In[ ]:


gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [9]
phi_list = [ 0.845 ]
total_dump_files = 101
inf_strain_max = 0.1

for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        
        gammadot = gammadotbeforepoint_list[i]*10**(-gammadot_list[j])
        filename_dist = 'd2min_' + str(gammadot) + 'pinwall.txt'
        filename_d2min = 'd2min_' + str(gammadot) + '.txt'

        with open(filename_dist, "rb") as fp:
            dist_dict = pickle.load(fp)
        with open(filename_d2min, "rb") as fq:
            d2min_dict = pickle.load(fq)
        
        strain_per_line = gammadot*0.01 # strain in one MD step
        #print(strain_per_line) 

        total_MDsteps = inf_strain_max/strain_per_line # total MD steps ran in between the first and last dump files
        #print(total_MDsteps)

        dump_frequency =  total_MDsteps/(total_dump_files-1) # how many MD steps are these dump files apart
        #print(dump_frequency)

        MDstep_list = [int(i*dump_frequency) for i in range(total_dump_files)] # list of MD steps of dump files
        #print(MDstep_list)
        
           
        for k in range(len(pin_list)):
            pin_num = pin_list[k]
            
            for l in range(len(phi_list)):
                phi_val = phi_list[l]
                
                # you can keep the next line or for specific MD step, give a specific value of n 
                for n in range(len(MDstep_list)-1): 
                    
                    ids = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['id']
                    distances_pin = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_pin_distance']
                    distances_wall = dist_dict[gammadot][pin_num][phi_val][MDstep_list[n]]['min_wall_distance']

                    fig, ax = plt.subplots() 

                    ax.set_ylabel('Min Distance From Pins', fontdict={'weight': 'bold'}, fontsize=20)
                    ax.set_xlabel('Min Distance From Wall', fontdict={'weight': 'bold'}, fontsize=20)

                    # Change the font size of tick labels
                    # ax.set_xticklabels(ax.get_xticks(), fontsize=15)  
                    # ax.set_yticklabels(ax.get_yticks(), fontsize=15)  

                    d2min = [] # initializing a list to catch d2min of midmove particless
                    for particle in ids:
                        d2min.append(d2min_dict[gammadot][pin_num][phi_val][MDstep_list[n]][particle])
                    
                    # color contour map
                    sc = ax.scatter(distances_wall, distances_pin, s=100, c=d2min, cmap='YlGnBu', marker='x', edgecolor='none') 

                    # Create a ScalarMappable object for the colorbar
                    cbar = plt.colorbar(sc, orientation='horizontal', extend='both')
                    cbar.ax.set_xlabel(r'$D^2_{min}$')

                    plt.rc('figure', figsize = (11, 6)) # to control size of figures
                    plt.legend()
                    plt.show()

