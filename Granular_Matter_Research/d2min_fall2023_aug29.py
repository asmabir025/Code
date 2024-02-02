#!/usr/bin/env python
# coding: utf-8

# For calculating d2min and saving it to a file, go to the block next to
# # Calculating d2min 
# 
# 
# ## The function path_name
# This function assumes your dump files or porfile files are in directories with specific structure. You can create this specific structure using the block next to the function dirpath.
# 
# # -------------------------------------------------------
# change June 19, 2023:  match t (m+1 conf; next) and t-Delta t (m conf; xr_ref) 
# 
# change June 20, 2023: add deformation (Delta r)^2 = D2min_i(K=0) (potentially normalization factor)
# 
# change June 21, 2023: (Delta r)^2 = D2min_i(K=1=unitaryMatrix)

# # Importing essential modules 

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


# # Setting styles

# In[2]:


mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (6, 5)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# # Defining functions for calculating D2min

# In[3]:


# you can modify path_name function. The function currently returns directory address
#of dump files on my local computer. 

def path_name(gammadot, pin_num, phi):
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
    dir_name += 'Phi0' + str(int(phi*1e5)) + '/'  #example: 1e-06/Pins9/Phi084500   
    dir_name += 'Dump_Files/'                  #example: 1e-06/Pins9/Phi084500/Dump_Files/
            
    path = os.path.join(cwd, dir_name)
    #example: [current directory]/1e-06/Pins9/Phi084500/Dump_Files/
    
    return path


# 
# # you can modify the next 5 lines. And the line with the variable rel_profile_dir.
# cwd = os.getcwd() 
# gammadotbeforepoint_list = [1] 
# gammadot_list = [6] 
# pin_list = [64] 
# phi_list = [0.835]
# 
# # Creating a structure of directories and subdirectories, for example, 
# # [current directory]/1e-06/Pins9/Phi084500/
# 
# for i in range(len(gammadotbeforepoint_list)): 
#     for j in range(len(gammadot_list)):
#     
#         # the name of the directory that is going to be created in that current directory. This new directory will
#         # contain different directories named as differnt shear rates. In the given structure, I kept the string
#         # empty. This means, all the directories named as different shear rates will be located in the current 
#         # directory, not under a directory within the current directory.
#         
#         rel_profile_dir = '' # you can write any string here.
#         gammadot = str(gammadotbeforepoint_list[i]*10**((-1)*gammadot_list[j]))
#         dir_profile = os.path.join(cwd, rel_profile_dir)
#         newdir_gammadot = os.path.join(dir_profile, gammadot) # example: [current directory]/1e-06
#         # creating the new directory. exist_ok=True leaves directory unaltered. 
#         os.makedirs(newdir_gammadot, exist_ok=True)
# 
#         for k in range(len(pin_list)):
# 
#             pin_dirname = 'Pins' + str(pin_list[k])
#             newdir_pin = os.path.join(newdir_gammadot, pin_dirname) # example: [current directory]/1e-06/Pins9
#             os.makedirs(newdir_pin, exist_ok=True)
# 
#             for l in range(len(phi_list)):
# 
#                 phi_dirname = 'Phi0' + str(int(phi_list[l]*1e5))
#                 newdir_phi = os.path.join(newdir_pin, phi_dirname) # example: [current directory]/1e-06/Pins9/Phi084500
#                 os.makedirs(newdir_phi, exist_ok=True)

# In[4]:


def system_info(path, ref_file):
    '''returns total particle number (Ntot) and length of box in x-direction from the reference file
    Unless total number of particles and length of box is changing over time, you should only use this function once.
    Parameters:
    path; str; the address of the directory containing dump files
    ref_file; str; reference confdump file to get Ntot, Lx
    '''
    ref_address = path + ref_file 
    open_file = open(ref_address, 'r')
    countline = 0

    # Reading in reference file to determine Ntot, Lx for the configuration
    
    for line in open_file.readlines(): # goes line by line in the confdump file
        countline +=1
        fields = line.split() 
        # as line is a string, fields is a list of strings separated by any space in line

        if countline == 4: #4th line of the confdump file
            Ntot = int(fields[0]) # 1st (only) element of 4th line
            
        elif countline == 6: # 6th line
            Lx = float(fields[1]) # 2nd (last) element of 6th line
            
            break # file-reading is done after line 6

    open_file.close()
    
    return Ntot, Lx


# In[5]:


def id_pos_list(Ntot, MD_step, path, file):
    '''extracts xy position and ID of each particle from a dump file
    Parameters:
    - Ntot; int; number of total particles 
    - MD_step; int; the MDstep number of the confdump file
    - path; str; full path of the directory containing dump files 
    - file; str; name of the confdump file
    Returns
    xy_pairs; dictionary; key: (int) particle ID and value: tuple (x,y)
    '''
    
    file = path + file
    
    xy_pairs = {} # initiating dictionary 
    
    with open(file) as csvfile:
        
        csvreader = csv.reader(csvfile, delimiter=' ') 
        for n in range(9): # skipping first 9 rows in the dump file
            next(csvreader)
            
        for row in csvreader: # row is a list of contents (string) in a row
        
            row = [x for x in row if x != '']

            xy_pairs[int(row[0])] = ( float(row[2]), float(row[3]) )
            # row[0]=particle ID, row[2]=x-position of the particle, row[3]=y-position
    
    return xy_pairs


# In[6]:


def neighbor(Lx, Ntot, xy_pairs):
    '''Saves neighbor information about each particle
    Parameters:
    Lx; int; x-length of system
    Ntot; int; total number of particles
    xy_pairs; dictionary; returned id_pos_list
    Returns:
    neigh_dict; dictionary; 
                key: particle ID (int), value: (sorted) lists of neighbor particle IDs.
    r_dict; dictionary; 
                key: particle ID (int); value: lists of tuples (x,y) of neighbor particles.
                                                Tuples sorted by neighbor IDs. 
    '''
    
    neigh_dict = {}
    r_dict = {}
    
    for i in range(1,Ntot+1):
        # ininitating empty list for each particle ID (key) in the dictionaries
        neigh_dict[i] = []
        r_dict[i] = []
        
    for i in range(1,Ntot+1):
        # getting particle position (x,y)
        xi = xy_pairs[i][0]
        yi = xy_pairs[i][1]
      
        for j in range(i+1, Ntot+1): 
            # we have already updated neighbors for IDs less than i in previous iterations
            # go through the full codes below; you will notice 
            
            # getting potential neighbor position (x,y)
            
            
            xj = xy_pairs[j][0]
            
            yj = xy_pairs[j][1]
            # getting the distance between a particle and a potential neighbor in both directions
            xr = xj-xi
            yr = yj-yi
            
            # periodic boundary conditions
            if xr > Lx/2: xr -= Lx 
            elif xr < - Lx/2: xr += Lx
                
            # distance squared
            rijto2 = xr * xr + yr * yr
            
            if rijto2 < 36: # distance < 6
                neigh_dict[i].append(j) # appending j to neighbor list of i
                neigh_dict[j].append(i) 
                # as i is also a neighbor to j; also this is updating the neighbors of j, who are less than j
                r_dict[i].append( (xr, yr) ) # appending j's distance from i 
                r_dict[j].append( (-xr, -yr) ) 
                # appending i's distance from j; updating the distances of neighbors of j, who are less than j
        
    return neigh_dict, r_dict


# In[7]:


def d2minAndDeltarto2(Ntot, Lx, neigh_dict, r_dict, xy_pairs_next):
    '''Calculates d2min for each particle at a certain MDstep
    Parameters:
    Lx; float; x-length of system
    Ntot; int; total number of particles
    xy_pairs_next; dictionary; returned id_pos_list (by using dump file of the next MD step )
    neigh_dict; dictionary; 
                key: particle ID (int), value: (sorted) lists of neighbor particle IDs.
    r_dict; dictionary; 
                key: particle ID (int); value: lists of tuples (x,y) of neighbor particles.
                                                Tuples sorted by neighbor IDs. 
    Returns:
    d2min_falk,Deltarto2
    d2min_falk; dictionary; key: particle ID (int), value: d2min (float)
    '''
    # X & Y matrices:
    Xmat_xx = np.zeros(Ntot)
    Xmat_xy = np.zeros(Ntot)
    Xmat_yx = np.zeros(Ntot)
    Xmat_yy = np.zeros(Ntot)
    Ymat_xx = np.zeros(Ntot)
    Ymat_xy = np.zeros(Ntot)
    Ymat_yx = np.zeros(Ntot)
    Ymat_yy = np.zeros(Ntot)
    Kmatxx = np.zeros(Ntot)
    Kmatxy = np.zeros(Ntot)
    Kmatyx = np.zeros(Ntot)
    Kmatyy = np.zeros(Ntot)
    
    
    r_dict_next = {} # initiating r_dict_next for the dump file from the next MD step
    for i in range(1, Ntot+1): 
        r_dict_next[i] = [] # initiating an empty list for each particle ID
         
    # This loop determines Xmat, Ymat, Kmat
    for i in range(1, Ntot+1):
        # getting position of particle i
        xi = xy_pairs_next[i][0]
        yi = xy_pairs_next[i][1]
        
        for k in range(len(neigh_dict[i])):
            j = neigh_dict[i][k]
            if j>i : 
                # we have already updated matrices for j<i neighbors in previous iterations
                # go through the full codes below; you will notice 
                
                # getting distance between particle and neighbor at reference MD step; 
                #     xj(t-Delta t)-xi(t-Delta t),yj(t-Delta t)-yi(t-Delta t)
                xr_ref = r_dict[i][k][0]
                yr_ref = r_dict[i][k][1]

                # getting position of neighbor j at next MD step; x(t),y(t)
                xj = xy_pairs_next[j][0]
                yj = xy_pairs_next[j][1]
                # getting distance between particle and neighbor in the next MD step; xj(t)-xi(t),yj(t)-yi(t)
                xr = xj-xi
                yr = yj-yi
                if xr > Lx/2: xr -= Lx
                elif xr < - Lx/2: xr += Lx
            
                # updating r_dict_next for both i and j
                r_dict_next[i].append( (xr, yr) )
                r_dict_next[j].append( (-xr, -yr) )
            
                # updating matrix elements of X and Y
                # i is the particle ID, i-1 is the index number;
                # list index starts from 0 but particle ID starts from 1
                Xmat_xx[i-1] += xr * xr_ref 
                Xmat_xy[i-1] += xr * yr_ref 
                Xmat_yx[i-1] += yr * xr_ref
                Xmat_yy[i-1] += yr * yr_ref
                Ymat_xx[i-1] += xr_ref * xr_ref
                Ymat_xy[i-1] += xr_ref * yr_ref
                Ymat_yx[i-1] += yr_ref * xr_ref
                Ymat_yy[i-1] += yr_ref * yr_ref

                Xmat_xx[j-1] += xr * xr_ref # for j neighbor, the added product is actually (-xr)*(-xr_ref)
                # same (-1)*(-1) for the rest
                Xmat_xy[j-1] += xr * yr_ref 
                Xmat_yx[j-1] += yr * xr_ref
                Xmat_yy[j-1] += yr * yr_ref
                Ymat_xx[j-1] += xr_ref * xr_ref
                Ymat_xy[j-1] += xr_ref * yr_ref
                Ymat_yx[j-1] += yr_ref * xr_ref
                Ymat_yy[j-1] += yr_ref * yr_ref
    for i in range(1, Ntot+1):
        # follow notes from Katharina Vollmayr-Lee and Falk and Langer Paper for d2min derivation
        onedivdetyi = 1/(Ymat_xx[i-1]*Ymat_yy[i-1] - Ymat_xy[i-1]*Ymat_yx[i-1])
        yinvxxi = Ymat_yy[i-1] * onedivdetyi
        yinvxyi = -Ymat_xy[i-1] * onedivdetyi
        yinvyxi = -Ymat_yx[i-1] * onedivdetyi
        yinvyyi = Ymat_xx[i-1] * onedivdetyi
        Kmatxx[i-1] = Xmat_xx[i-1] * yinvxxi + Xmat_xy[i-1] * yinvyxi
        Kmatxy[i-1] = Xmat_xx[i-1] * yinvxyi + Xmat_xy[i-1] * yinvyyi
        Kmatyx[i-1] = Xmat_yx[i-1] * yinvxxi + Xmat_yy[i-1] * yinvyxi
        Kmatyy[i-1] = Xmat_yx[i-1] * yinvxyi + Xmat_yy[i-1] * yinvyyi
    
    d2min_falk = {} # initiating a dictionary to catch d2min against the particle ID as key
    Deltarto2 = {}

    # This loop calculates D2min: 
    for i in range(1,Ntot+1):
        d2min_falk[i] = 0
        Deltarto2[i] = 0
        for k in range(len(neigh_dict[i])): # iterating over all neighbors of i
            
            xr_ref = r_dict[i][k][0]
            yr_ref = r_dict[i][k][1]
            xr = r_dict_next[i][k][0]
            yr = r_dict_next[i][k][1]

            tnuxi = xr - Kmatxx[i-1] * xr_ref - Kmatxy[i-1] * yr_ref
            tnuyi = yr - Kmatyx[i-1] * xr_ref - Kmatyy[i-1] * yr_ref
            d2min_falk[i] += tnuxi * tnuxi + tnuyi * tnuyi
            
            tnuxiK1 = xr - xr_ref
            tnuyiK1 = yr - yr_ref
            Deltarto2[i] += tnuxiK1 * tnuxiK1 + tnuyiK1 * tnuyiK1


#     d2min_mean = round_to_6(sum(d2min_falk.values()) / len(d2min_falk))
    
    return d2min_falk,Deltarto2


# file = open( "d2min_1e-6.txt", "rb+") #write-binary format
# 
# d2min_extra = pickle.load(file) 
# 
# file.close()

# # Calculating d2min

# In[14]:


''' Only next few lines you might need to change 
'''
    
gammadotbeforepoint_list = [1]
gammadot_list = [4] 
pin_list = [16]
phi_list = [0.845]
total_dump_files = 11 # Number of dump files we are analyzing #usually 101, here for testing fewer
inf_strain_min = 0.5004
inf_strain_max = 0.5274  #usually 0.1 here for testing because 11 instead of 101 # total amount of strain occured in these dump files

# this could be any MD step; we are only using this variable for getting the information of Ntot and Lx, 
# which do not change over MD steps according to the current setting
file_str = 'confdumpallinfMD' # could be 'confdumpallelastic' or anything you name for the dump files


# Do not change the sequence of codes from here on
for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):

        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j])) # shear rate
        ref_MD_step = round(inf_strain_min / (gammadot*0.01) )
        ref_file = f'{file_str}{ref_MD_step}.data'
        
        d2min_dict = {}
        d2min_dict[gammadot] = {} # initiating empty dictionary against key of gammadot
        Deltar_dict = {}     #potential future normalization: distances as is, i.e. including affine part
        Deltar_dict[gammadot] = {}
        
        strain_per_line = gammadot*0.01 # strain in one MD step
        

        print(strain_per_line) 

        total_MDsteps = round((inf_strain_max - inf_strain_min)/strain_per_line) # total MD steps ran in between the first and last dump files
        print(total_MDsteps)

        dump_frequency =  round(total_MDsteps/(total_dump_files-1)) # how many MD steps are these dump files apart?
        print(dump_frequency)

        MDstep_list = [int(ref_MD_step + iMDl*dump_frequency) for iMDl in range(total_dump_files)] # list of MD steps of dump files
        print(MDstep_list)

        for k in range(len(pin_list)):

            pin_num = pin_list[k]
            d2min_dict[gammadot][pin_num]={} # initiating empty dictionary against key of pin_num
            Deltar_dict[gammadot][pin_num]={}

            for l in range(len(phi_list)):

                phi_val = phi_list[l]
                path = path_name(gammadot, pin_num, phi_val)
                # go-to-function: path_name
                Ntot, Lx = system_info(path, ref_file)
                # go-to-function: system_info
                d2min_dict[gammadot][pin_num][phi_val]={} # initiating empty dictionary against key of phi_val
                Deltar_dict[gammadot][pin_num][phi_val]={}

                for m in range(len(MDstep_list)-1):
                    
                    file = f'{file_str}{MDstep_list[m]}.data'
                    xy_pairs0 = id_pos_list(Ntot, MDstep_list[m], path, file)
                    # go-to-function: id_pos_list
                    
                    neigh_dict, r_dict = neighbor(Lx, Ntot, xy_pairs0)
                    # go-to-function: neighbor
                    
                    file = f'{file_str}{MDstep_list[m+1]}.data'
                    xy_pairs_next = id_pos_list(Ntot, MDstep_list[m+1], path, file)
                    # go-to-function: id_pos_list
                    
                    # updating d2min_dict for specific MDstep
                    d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]],\
                     Deltar_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]] = \
                                                   d2minAndDeltarto2(Ntot, Lx, neigh_dict, r_dict, xy_pairs_next)
                    
#                     # add following lines for D2min testing:
#                     filed2mintest = 'd2minviapython_' + str(m+1)
#                     fileoutd2mintest = open(filed2mintest,mode='w')
#                     for iD2 in range(1,Ntot+1):
#                         print(iD2," ",d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]][iD2],file=fileoutd2mintest)
#                     fileoutd2mintest.close()
#                     # add following lines for Deltar^2 testing:
#                     fileDeltarto2test = 'Deltarto2viapython_' + str(m+1)
#                     fileoutDeltarto2test = open(fileDeltarto2test,mode='w')
#                     for iD2 in range(1,Ntot+1):
#                         print(iD2," ",Deltar_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]][iD2],file=fileoutDeltarto2test)
#                     fileoutDeltarto2test.close()
        
        # saving d2min_dict for specific shear rate
        newfile = 'd2min_' + str(gammadot) + '.txt'  
        with open(newfile, 'wb') as f:
            pickle.dump(d2min_dict, f, pickle.HIGHEST_PROTOCOL)  

        # saving Deltar_dict for specific shear rate
        newfile2 = 'Deltar_' + str(gammadot) + '.txt'  
        with open(newfile2, 'wb') as f2:
            pickle.dump(Deltar_dict, f2, pickle.HIGHEST_PROTOCOL)


# In[15]:


for shear_rate in d2min_dict:
    for pin in d2min_dict[shear_rate]:
        for phi in d2min_dict[shear_rate][pin]:
            num_events = 0
            for mdstep in d2min_dict[shear_rate][pin][phi]:
                d2min_mdstep = d2min_dict[shear_rate][pin][phi][mdstep]
                for particle in d2min_mdstep:
                    if d2min_mdstep[particle]>1:
                        num_events += 1
                       
                print(num_events)
                    
                


# In[ ]:




