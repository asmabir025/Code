#!/usr/bin/env python
# coding: utf-8

# #For plotting histogram of displacement(s) to get the idea of delta t, go to the block next to
# # Plotting  histogram of displacement
# after running each code block before that.
# 
# 
# # -------------------------------------------------------

# In[ ]:


import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy as sc
from scipy import stats
import statistics as st
import pickle
import os
import glob


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (5, 5)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# In[ ]:


def path_name(gammadot, pin_num, phi):
    '''
    creates directory names that are in my home directory and contain the data for aps meeting 
    Parameters:
        - cwd (current working directory); str 
        - gammadot; str
        - pin_num; int
        - phi; float
    returns path; str
            
    '''
    cwd = os.getcwd()
    dir_name =  str(gammadot) + '/'                                             #example: 1e-06/
    dir_name += 'Pins' + str(pin_num) + '/'                                #example: 1e-06/Pins0
    dir_name += 'Phi0' + str(int(phi*1e5)) + '/'                        #example: 1e-06/Pins0/Phi083500
    dir_name += 'Dump_Files/'                                     #example: 1e-06/Pins0/Phi083500/Dump_Files
                    
    
    path = os.path.join(cwd, dir_name)
    
    return path


# In[ ]:


def read_data_wall(pathname):
    '''
    reads Lx, and the id, type columns of the wall dump file
    parameters:
    pathname; path to the dump files; str
    returns 
    idarraywall; array; ids of wall particles
    typearraywall; array; types of wall particles
    Lx; float; length of the system
    '''
    global wall_file
    
    path_str = os.path.join(pathname, wall_file)
    file = glob.glob(path_str)
    idarraywall, typearraywall = np.loadtxt(file[0], skiprows=9, usecols = (0,1), dtype='int', unpack=True)

    open_file = open(file[0], 'r')
    countline = 0
    for line in open_file.readlines():
        countline +=1
        fields = line.split()

        if countline == 6:
            Lx = float(fields[1]) # reading the length of mid region
            break
    open_file.close()

    return idarraywall, typearraywall, Lx


# In[ ]:


def array_nowall(idarraywall, MD):
    '''
    removes the wall particle information from the arrays 
    paramter:
    arrays returned by read_data_wall function 
    returns
    midmove_id; array; ids of midparticles
    midmove_type; array; types of midparticles
    midmove_xu; array; unwrapped x-coordinates of midparticles
    midmove_yu; array; unwrapped y-coordinates of midparticles
    '''
    global file_str, path
    
    # reading all particle properties from a dump file
    file = path + f'{file_str}MD{MD}.data'
    idarray, typearray, xuarray, yuarray = np.loadtxt(file, skiprows=9, usecols = (0,1,8,9), dtype='float', unpack=True)
   
    # getting wall and pin indices
    idwall = [idarraywall[i]-1 for i in range(len(idarraywall))]
    type_pin_indices = np.where(typearray == 3)[0] # indices of pin particles in type array
    indices_removed = np.concatenate((idwall, type_pin_indices)) # indices of walls and pins

    # getting indices of midmove particles
    midmove_id = np.delete(idarray, indices_removed)
    midmove_type = np.delete(typearray, indices_removed)
    midmove_xu = np.delete(xuarray, indices_removed)
    midmove_yu = np.delete(yuarray, indices_removed)
    
    return midmove_id, midmove_type, midmove_xu, midmove_yu


# In[ ]:


def pos_dict(midmove_id, midmove_xu, midmove_yu):
    '''
    saves x,y information information of all particles in a dictionary
    paramters:
    arrays returned by array_nowall() except midmove_type
    returns 
    pos_dict; dictionary
            keys: particle IDs; int
            values: tuples (xu, yu)
    '''
    
    pos_dict = {} # initializing a dictionary 

    for i in range(len(midmove_id)):
        pos_dict[midmove_id[i]] = (midmove_xu[i], midmove_yu[i])
        
    return pos_dict


# In[ ]:


def displacement(pos_dict1, pos_dict2):
    '''
    parameter:
    pos_dict1, pos_dict2 : two dictionaries of tuple (xu, yu) from two different MD steps using pos_dict function
    returns
    delta_pos; dictionary
               keys: particle ID; int
               values: tuples of displacements in x, y directions and non-affine component in x direction
    '''
    global dump_frequency, Lx, gammadot
    
    vel_wall = -1* gammadot*0.01*Lx*0.5 # lower wall velocity 
    slope_aff = (-2*vel_wall)/Lx # '-' ensures positive slope of velocity profile
    
    delta_pos = {} # initializing an empty dictionary
    
    for i in pos_dict1: # i = id of a particle 

        delta_x_i = pos_dict2[i][0] - pos_dict1[i][0] # calculating difference in x-positions from two MD steps
        delta_y_i = pos_dict2[i][1] - pos_dict1[i][1] # calculating difference in y-positions from two MD steps
        
        vel_i = vel_wall + slope_aff*pos_dict1[i][1] # calculating affine velocity of the particle
        delta_x_nonaff_i = delta_x_i - vel_i*dump_frequency*0.01 # calculating non-affine component of x-displacement
        
        # saving displacements (x, y, nonaffine-x) for the particle
        delta_pos[i] = (delta_x_i, delta_y_i, delta_x_nonaff_i) 
        
    return delta_pos


# In[ ]:


def sorted_xy_lists(delta_pos):
    '''creates three sorted lists of tuples; sorting based on delta_x, delta_y, and delta_x_nonaff, respectively

    Parameters:
    delta_pos: a dictionary returned by displcement function above
    
    Returns three sorted lists of tuples: 
    For each tuple in any list:
    - First element: particle ID
    - Second element is a tuple of delta_x, delta_y, and delta_x_nonaff
    '''

    # Before proceeding to the next block of code:
    # The dictionary.items() returns a list of tuples, 
    # where the first value of each tuple is the key, and the second value is the corresponding value.
    # Sorting is performed using lambda functions.
    # In the lambda functions, 'y' represents the output, and 'x' represents the input.
    # Everything after ':' provides details of the function.
    # For example, in the first lambda function of the code, the output is obtained from k[1][0].
    # Here, 'k' refers to each tuple in delta_pos.items().
    # k[1] represents the second element of the tuple, which is a value in delta_pos.
    # So, k[1] is a tuple itself, containing delta_x_i, delta_y_i, and delta_x_nonaff_i.
    # Therefore, k[1][0] corresponds to delta_x_i.

    # Sorting the displacement dictionary based on displacement in x, y, and non-affine x, 
    # respectively
    sorted_jump_x = sorted(delta_pos.items(), key=lambda k: k[1][0])
    sorted_jump_y = sorted(delta_pos.items(), key=lambda k: k[1][1])
    sorted_jump_x_nonaff = sorted(delta_pos.items(), key=lambda k: k[1][2])

    return sorted_jump_x, sorted_jump_y, sorted_jump_x_nonaff


# In[ ]:


def binning():
    '''
    Creates arrays for displacement and occurrences to facilitate histogram plotting.

    Returns:
    delta_ar_bin: Array of displacements after binning.
    occurrences: Array of the number of occurrences for displacements in delta_ar_bin.
    data_displacement: Array of raw displacement data.
    even_displacement: Array of evenly spaced displacements based on the lowest and highest displacements.
    '''

    # Global variables
    global param, num_bins, sorted_jump_x, sorted_jump_y, sorted_jump_x_nonaff
    
    # Selecting the appropriate sorted_tuple_list based on the parameter value
    if param == 0:
        sorted_tuple_list = sorted_jump_x
    elif param == 1:
        sorted_tuple_list = sorted_jump_y
    elif param == 2:
        sorted_tuple_list = sorted_jump_x_nonaff

    # Finding the smallest and largest displacements from the sorted tuple list
    low = sorted_tuple_list[0][1][param]
    high = sorted_tuple_list[-1][1][param]

    # Creating an evenly spaced displacement array within the interval of low and high
    even_displacement = np.linspace(low, high, num_bins)

    # Creating a displacement array from the sorted_tuple_list
    data_displacement = np.array([sorted_tuple_list[i][1][param] for i in range(len(sorted_tuple_list))])

    # Initializing arrays
    occurrences = np.array([])
    delta_ar_bin = np.array([])

    for i in range(len(even_displacement)-1):
        # Binning the displacements
        intermediate = np.where((even_displacement[i] <= data_displacement) & (data_displacement <= even_displacement[i+1]))[0]
        occurrences = np.append(occurrences, len(intermediate))

        for j in range(len(intermediate)):
            delta_ar_bin = np.append(delta_ar_bin, (even_displacement[i]+even_displacement[i+1])/2)

    return delta_ar_bin, occurrences, data_displacement, even_displacement


# In[ ]:


def plot_dist():
    '''
    plots displacement distribution, adds mean, median, and mean +/- standard deviation lines
    parameters:
        delta_ar; array
        bin_ar; array
        occurences; array
        coord_str; string (either 'x', 'y', or 'x (non-affine)')
        MD_diff; string
    '''
    global dump_frequency, delta_ar_bin, occurences, data_displacement, param, even_displacement
    
    if param == 0:
        coord_str = 'x'
    elif param == 1:
        coord_str = 'y'
    else:
        coord_str = r'x_{NA}'
    
    mean_ar = np.mean(data_displacement)
    # median_ar = np.median(delta_ar)
    std_ar = np.std(data_displacement)

    # plt.hist(delta_bin_ar)
    fig = plt.figure()
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    plt.stairs(occurences, even_displacement)
    plt.xlabel(f'$< \Delta {coord_str} >$') 
    plt.xlim(-0.4, 0.4) # you might need to change
    plt.ylabel('Occurence')
    #plt.title('$\Delta$'+coord_str + ' distribution for MD step difference ' + MD_diff)
    
    
    plt.axvline(mean_ar, c='g', label = f'$< \Delta {coord_str} >$')
    # plt.axvline(median_ar, c='g', label = 'median')
    plt.axvline(mean_ar + 3*std_ar, c='cyan', label = f'$< \Delta {coord_str} > + 3\sigma$')
    plt.axvline(mean_ar - 3*std_ar, c ='violet', label = f'$< \Delta {coord_str} > - 3\sigma$')
    plt.legend(loc = 'upper left', prop={'size': 8})
    plt.savefig('distribution_nonaff.png')


# In[ ]:


def jump_ids(sorted_jump_y, sorted_jump_x_nonaff):
    '''
    Parameters:
    sorted_jump_y, sorted_jump_x_nonaff; both returned by sorted_xy_lists function
    returns the ids of particles with exceptionally high displacements in y and nonaffine-x
    
    '''
    # getting displacements in particular directions from the sorted lists
    sorted_y = [sorted_jump_y[i][1][1] for i in range(len(sorted_jump_y))] 
    sorted_x = [sorted_jump_x_nonaff[i][1][2] for i in range(len(sorted_jump_x_nonaff))]
    
    # calculating mean and standard deviation of displacements in each direction
    mean_y = st.mean(sorted_y)
    mean_x = st.mean(sorted_x)
    std_x = st.stdev(sorted_x) 
    std_y = st.stdev(sorted_y)
    
    # getting indices in the sorted_y and sorted_x, respectively, 
    #where displacement is outside mean +/3*sigma
    # '|' represents logical 'or'
    high_y_indices = np.where((sorted_y > 3*std_y + mean_y) | (sorted_y < -3*std_y + mean_y))[0]
    high_x_indices = np.where((sorted_x > 3*std_x + mean_x) | (sorted_x < -3*std_x + mean_x))[0]
    
    # getting particle ids with exceptional displacement in sets
    jump_y_id = {sorted_jump_y[i][0] for i in high_y_indices}
    jump_x_id = {sorted_jump_x_nonaff[i][0] for i in high_x_indices}
    
    # combining both sets
    jump_id = jump_x_id.union(jump_y_id)
    
    return jump_id


# # Plotting histogram of displacement

# In[ ]:


''' Only next few lines you might need to change 
'''
gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [9]
phi_list = [0.845]
total_dump_files = 101 # Number of dump files we are analyzing 
inf_strain_max = 0.1  # total amount of strain occured in these dump files
# scale determines delta t. delta t = scale* dump_frequency * 0.01; where 
# dump_frequency is the difference in many MD steps between two adjacent dump files
scale = 1
param = 0 # val represents whether we want to sort delta_x (param = 0), delta_y (1), or delta_x_nonaff (2)
num_bins = 1000 # number of bins in the histogram
wall_file = 'wallpartidtype' # file name which contains just wall information
file_str = 'confdumpallinf' # could be 'confdumpallelastic' or anything you name for the dump files


# initiating a dictionary to catch d2min values. Keys: MDstep values 
# not to be confused with d2min_dict, which reads the d2min dictionary later from a file
d2min_vals = {} 

for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):

        gammadot = gammadotbeforepoint_list[i]* 10**(-gammadot_list[j])
        # print(gammadot)
        strain_per_line = gammadot*0.01 # strain in one MD step
        #print(strain_per_line) 

        total_MDsteps = inf_strain_max/strain_per_line # total MD steps ran in between the first and last dump files
        #print(total_MDsteps)

        dump_frequency =  total_MDsteps/(total_dump_files-1) # how many MD steps are these dump files apart
        #print(dump_frequency)

        MDstep_list = [int(i*dump_frequency) for i in range(total_dump_files)] # list of MD steps of dump files
        #print(MDstep_list)
        
        # reading d2min from a file 
        file = 'd2min_' + str(gammadot) + '.txt'
        with open(file, "rb") as fp:
                d2min_dict = pickle.load(fp)

                
        for k in range(len(pin_list)):
            pin_num = pin_list[k]
        
            for l in range(len(phi_list)):

                phi_val = phi_list[l]
                # getting path to the dump files
                path = path_name(gammadot, pin_num, phi_val)
                # getting information about wall
                idwall, typewall, Lx = read_data_wall(path)
            

                for m in range(MDstep_list):
                    d2min_vals[MDstep_list[m]] = []
                    
                    # getting properties of midmove particles from two neighbor MDsteps
                    midmove_id1, midmove_type1, midmove_xu1, midmove_yu1 = array_nowall(idwall, MDstep_list[m])
                    midmove_id2, midmove_type2, midmove_xu2, midmove_yu2 = array_nowall(idwall, MDstep_list[m+scale])
                    
                    # creating dictionaries with values of position coordinates of all particles 
                    # from two neighbor MDsteps
                    pos_dict1 = pos_dict( midmove_id1, midmove_xu1, midmove_yu1 )
                    pos_dict2 = pos_dict( midmove_id2, midmove_xu2, midmove_yu2 )
                    
                    # creating dictionaries of displacements and then lists of sorted displacements 
                    # in a particular direction
                    delta_pos = displacement(pos_dict1, pos_dict2)
                    sorted_jump_x, sorted_jump_y, sorted_jump_x_nonaff = sorted_xy_lists(delta_pos)
        
                    # binning and plotting
                    delta_ar_bin, occurences, data_displacement, even_displacement = binning()
                    plot_dist()
                    
                    # saving particle IDs with exceptional displacement
                    jump_id = jump_ids(sorted_jump_y, sorted_jump_x_nonaff)
                    # also getting d2min values of these exceptional particles       
                    for ids in jump_id:
                        d2min_vals[MDstep_list[m]].append(d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m]][ids])
                        
                    



    


# # I used the next block of code to create a plot to help understand the time scale of d2min for APS March Meeting Presentation

# In[ ]:


d2min_list = list(d2min_val_dict.values())[0] # list of high d2min values 
num_list = [i for i in range(len(d2min_list))]
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)

plt.scatter(num_list, d2min_list ,color='r', marker = 'x')

plt.xticks([])
plt.xlabel('Individual Particles')
plt.ylabel(r'$D^2_{min}$')

plt.axhline(1, color = 'k', label = r'$D^2_{min} = 1$')
plt.legend()
#plt.axhline(min(d2min_list))
plt.savefig('d2min_higher.png')


# In[ ]:




