#!/usr/bin/env python
# coding: utf-8

# For creating average velocity profiles right away,  go to the block next to 
# # Creating average velocity profile...
# ### You also need to run the previous blocks 
# 
#  For curve-fitting average profile with Brian Tighe's mode, go to the block next to
# # Curve-fitting vx vs y...
# ### You also need to run previous blocks except the block next to Creating average velocity profile...
# 
# For creating animation of profiles, go to the block next to
# # Creating animation of profiles
# ### You also need to run the blocks from the beginning till (excluding) Creating average velocity profile...
# 
# For creating correlation plots, go to the block next to
# # Creating correlation plot
# ### You may also need to run the blocks from the beginning till (including) Creating average velocity profile...
# 
# 
# 
# ## The function dirpath
# This function assumes your dump files or porfile files are in directories with specific structure. You can create this specific structure using the block next to the function dirpath.
# 
# To better understand the code for each purpose, go to the specified block. Check the functions sequentially and read the specific function one at a time. You can always print out stuffs or play with small part of the codes!
# 
# # ------------------------------------------------------------------------

# # Importing essential modules 

# In[ ]:


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


# # Setting styles

# In[ ]:


mpl.style.use('seaborn-talk') 
plt.rc('figure', figsize = (6, 5)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size
plt.rcParams['text.usetex'] = True


# # Defining functions for creating velocity profiles

# In[ ]:


def dirpath(gammadot, pin_num, phi, filename):
    '''
    creates full path to the file 
    Parameters:
        - cwd (current working directory); str 
        - gammadot; str
        - pin_num; int
        - phi; float
        - filename; str
    returns path; str
            
    '''
    cwd = os.getcwd()                          #getting current directory
    dir_name =  str(gammadot) + '/'            #example: 1e-06/
    dir_name += 'Pins' + str(pin_num) + '/'    #example: 1e-06/Pins9
    dir_name += 'Phi0' + str(int(phi*1e5)) + '/'  #example: 1e-06/Pins9/Phi084500
    file_path = dir_name + filename            #example: 1e-06/Pins9/Phi084500/profileinf
    
    path = os.path.join(cwd, file_path)        #example: [current directory]/1e-06/Pins9/Phi084500/profileinf
    
    return path
    


# ### Creating specific structured directories and subdirectories
# To use the next code block change it from markdown to code

# In[ ]:


# you can modify the next 5 lines. And the line with the variable rel_profile_dir.
cwd = os.getcwd() # getting the current directory
gammadotbeforepoint_list = [1] 
gammadot_list = [6] 
pin_list = [64] 
phi_list = [0.835]

# Creating a structure of directories and subdirectories, for example, 
# [current directory]/1e-06/Pins9/Phi084500/

for i in range(len(gammadotbeforepoint_list)): 
    for j in range(len(gammadot_list)):
    
        # the name of the directory that is going to be created in that current directory. This new directory will
        # contain different directories named as differnt shear rates. In the given structure, I kept the string
        # empty. This means, all the directories named as different shear rates will be located in the current 
        # directory, not under a directory within the current directory.
        
        rel_profile_dir = '' # you can write any string here.
        gammadot = str(gammadotbeforepoint_list[i]*10**((-1)*gammadot_list[j]))
        dir_profile = os.path.join(cwd, rel_profile_dir)
        newdir_gammadot = os.path.join(dir_profile, gammadot) # example: [current directory]/1e-06
        # creating the new directory. exist_ok=True leaves directory unaltered. 
        os.makedirs(newdir_gammadot, exist_ok=True)

        for k in range(len(pin_list)):

            pin_dirname = 'Pins' + str(pin_list[k])
            newdir_pin = os.path.join(newdir_gammadot, pin_dirname) # example: [current directory]/1e-06/Pins9
            os.makedirs(newdir_pin, exist_ok=True)

            for l in range(len(phi_list)):

                phi_dirname = 'Phi0' + str(int(phi_list[l]*1e5))
                newdir_phi = os.path.join(newdir_pin, phi_dirname) # example: [current directory]/1e-06/Pins9/Phi084500
                os.makedirs(newdir_phi, exist_ok=True)


# In[ ]:


def system_info(gammadot_val, file):
    '''returns 
    - the list of MD steps
    - a dictionary containing vx, vy, sigmaxy values as a nested dictionary
    For example, 
    profile_dict = {1: {'y': [5, 5, 5,..], 'vx': [1, 1.5, 2,..], 'vy':[.....], 'sigmaxy':[....] }, 
                    2:{...}, 
                    3:{...},
                    ...} 
    Here, 1 is the first layer (closest to the bottom wall). Lists of 'y', 'vx',.. are y, vx,... values
    of the layer over time
    
    parameters:
    -file; str 
    
    '''
   
    layer = 32 # Number of layers 
    profile_dict = {}  # initiating an empty dictionary
    for i in range(1, layer+1):
        profile_dict[i] = {}       # initiating an empty dictionary for each key (layer)      
        
        # initiating empty lists for each necessary key in the nested dictionary:
        profile_dict[i]['vx'] = []
        profile_dict[i]['vy'] = []
        profile_dict[i]['y'] = []
        profile_dict[i]['sigmaxy'] = []
        
     
    mdstep = [] # initiating a list to catch mdstep values for each profile (snapshot)                  
                        
    with open(file, 'r') as the_file: 

        for line in the_file.readlines(): # line is a string 
            
            fields = line.split()  # creating a list of items separated by any amount of space in line 
            if len(fields) == 3:   # you can look at porofileinf and find that lines with 3 three values
                                   # contain Timestep (1st), Number-of-chunks (2nd), Total-count of atoms (3rd)
                    
                mdstep.append(int(fields[0])) # appending as integers as fields is a list of strings

            if len(fields) == 7:   # these are the lines with informations about profile quantities

                layer_num = int(fields[0]) # layer ID
                y = float(fields[2])       # y-coordinate of middle of the layer
                vx = float(fields[4])      # average velocity of all particles in the layer of the layer at that MDstep 
                sigma_xy = float(fields[6]) # average shear stress ......

                # saving the quantities of each layer at the mdstep
                profile_dict[layer_num]['vx'].append( vx ) 
                profile_dict[layer_num]['y'].append( y )
                profile_dict[layer_num]['sigmaxy'].append( sigma_xy )

    Lx = 2*float(fields[1]) # x-coordinate of the middle of any layer is Lmid
    vbottom = - Lx/2*gammadot_val # calculating velocity of the bottom wall
    
    return Lx, vbottom, mdstep, profile_dict


# In[ ]:


def affine_profile(vbottom, Lx, lywall, profile_dict):
    '''returns 
    - y-coordinates of the layers according to ascending sort 
    - affine velocities (vx) of each layer
    Parameters:
    - vbottom; velocity of the bottom wall
    - Lx; float; x-length of the system
    - lywall; float; thickness of either side of the wall
    - profile_dict; the dictionary returned from the function system_info
    '''
    slope = -2*vbottom/Lx # slope of the affine vx vs y plot
    # initiating lists catch y-coordinates and affine velocities
    y_coords = []
    vx_affine = []
    
    for key in profile_dict: # key is the layer ID
        y_coords.append( profile_dict[key]['y'][0] - lywall) # shifting y-coordinates towards zero
        
    y_coords.sort() # Dictionary keys are not always sorted; normally they are sorting according to insertion order 
    
    for y in y_coords: 
        vx_affine.append(slope*y + vbottom) # simply y = mx + c 
        
    return y_coords, vx_affine


# In[ ]:


def stats_prof(profile_dict, quantity_str, Lx):
    '''Returns two dictionaries; each have same keys. The keys are the y-coordinates of the layers.
    - average velocity of each layer across all snapshots against its key
    - uncertainties in velocity of each layer across all snapshots against its key
    Parameter:
    - profile_dict; a dictionary (returned by system_info function)
    - quantity_str; string; for example, 'vx'
    - Lx; int; x-length of the system
    '''
    avg_profile = {} # initiating a dictionary catch average velocity for each layer 
    error_profile = {} # initiating a dictionary catch uncertainty of velocity for each layer 
    
    for layer in profile_dict: # layer is the layer ID
        # once again; keys of both dictionaries are the y-coordinates of the layers
        avg_profile[profile_dict[layer]['y'][0]- lywall ] = stats.mean(profile_dict[layer][quantity_str])
        error_profile[profile_dict[layer]['y'][0]- lywall] = stats.stdev(profile_dict[layer][quantity_str])/np.sqrt(len(profile_dict[layer][quantity_str]))
        # uncertainty = standard deviation/sqrt(number of samples)
        
    return avg_profile, error_profile


# # Creating average velocity profile from all snapshots

# In[ ]:


'''Only next few lines you might need to change 
''' 
gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [0,9,16,36,64]
phi_list = [0.845] 
lywall = 3*2.8 # thickness of either side of the wall
file_name = 'profileinf'
quantity_str = 'vx'

# styles
color_list = ['blue', 'red', 'cyan', 'violet', 'pink'] 
# change the color list if the number of pin combinations is more than 5
fig = plt.figure()
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)

# Do not change the sequence of codes from here on

for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j])) # shear rate value
     
        for k in range(len(pin_list)):
            pin_num = pin_list[k] 
            
            for l in range(len(phi_list)):
                phi_val = phi_list[l]
            
                path_file = dirpath(gammadot, pin_num, phi_val, file_name) 
                # or you can give path manually; i.e. without using function, if you are testing only ony combination
                # go-to-function: path_file

                Lx, vbottom, mdstep_list, profile_dict = system_info(gammadot, path_file)
                # go-to-function: system_info
                
                y_coords, vx = affine_profile(vbottom, Lx, lywall, profile_dict)
                # go-to-function: affine_profile

                avg_prof, error_prof = stats_prof(profile_dict, quantity_str, Lx)
                # go-to-function: stats_prof

                plt.scatter(avg_prof.keys(), avg_prof.values(), color=color_list[k], marker = '.', label = f'Npin = {pin_num}')
     
        # notice the indentation; currently we are plotting only one phi case
        # If there are multiple phi cases, you might want to move next 5-7 lines to one tab right
        plt.xlabel(r'$y$')
        plt.ylabel(r'$v_x$', fontsize=20)
        plt.yticks(fontsize=15)
        #plt.ylim(-0.003, 0.003)
        
        plt.plot(y_coords, vx) # plotting affine line
        
        plt.legend(loc = 'upper left')
        
        '''You might want to change the next line
        '''
        img_name = f'plot_profile_{gammadot}_inf' # naming saved file
        plt.savefig(img_name)
        # plt.clf()
                
                


# In[ ]:


def velx_fit(y, k, G):
    '''Brian Tighe's fitting function
    Parameters:
    - y; float; y-coordinates of layers
    - k; float; spring constant
    - G; float; shear modulus
    '''
    global gammadot
    global Lx
    return gammadot*Lx/2*np.sinh(np.sqrt(k/G)*y)/np.sinh(np.sqrt(k/G)*(Lx/2))


# In[ ]:


def curve_fit():
    '''curve fits using velx_fit function
    returns 
    - k; spring constant
    - uncertainty in k
    - G; shear modulus
    - uncertainty in G
    '''
    global avg_prof
    global error_prof
    global Lx
    
    xdata = list(avg_prof.keys())
    xdata = [xdata[i] - Lx/2 for i in range(len(xdata))] # shifting by -Lx/2; Brian Tighe's function requires this
    ydata = list(avg_prof.values())
    udata = list(error_prof.values())

    
    popt, pcov = optimize.curve_fit(velx_fit,xdata,ydata,sigma=udata, p0=[ 0.1, 0.02 ], absolute_sigma=True)
    # p0: list of initial guesses; you need to play around these values to get better estimates
    # G = 0.02-04; according to the calculations of G done prior to APS March Meeting 2023
    
    return popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1][1])


# # Curve-fitting vx vs y according to Brian Tighe's Model

# In[ ]:


'''Only next few lines you might need to change 
''' 
# I would suggest first playing with only one combination of gammadot, pin, and phi
gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [64]
phi_list = [0.845]
lywall = 3*2.8  # thickness of either side of the wall
file_name = 'profileinf'
quantity_str = 'vx'

# initating lists to catch k,G, uncertainties for each combination of Npin 
k_list = []
delta_k_list = []
delta_G_list = []
G_list = []


color_list = ['blue', 'red', 'cyan', 'violet', 'pink']
# change the color list if the number of pin combinations is more than 5

# Do not change the sequence of codes from here on
for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        
        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j])) # calculating shear rate
        

        for k in range(len(pin_list)):
            pin_num = pin_list[k]
            for l in range(len(phi_list)):
                phi_val = phi_list[l]
                
                fig = plt.figure() # creating new figure 
                
                # styles
                fig.patch.set_facecolor('none')
                fig.patch.set_alpha(0)


                path_file = dirpath(gammadot, pin_num, phi_val, file_name) # you can also give path manually
                # go-to-function: dirpath 

                Lx, vbottom, mdstep_list, profile_dict = system_info(gammadot, path_file)
                # go-to-function: system_info

                y_coords, vx = affine_profile(vbottom, Lx, lywall, profile_dict)
                # go-to-function: affine_profile

                avg_prof, error_prof = stats_prof(profile_dict, quantity_str, Lx)
                # go-to-function:  stats_prof

                k0, delta_k, G, delta_G = curve_fit()
                # go-to-function: curve_fit, velx_fit
                k_list.append(k0)
                G_list.append(G)
                delta_k_list.append(delta_k)
                delta_G_list.append(delta_G)
                print('k:',k0, 'delta k:',delta_k, 'G:', G, 'delta G:', delta_G)


                y_list = np.linspace(-Lx/2, Lx/2, 1000) # creating a list of more 'continuous' y coordinates 
                predicted_vx_3 = [velx_fit(y, k0, G) for y in y_list] 
                # creating predicted velocites for these coordinates
                # plotting vx vs y according to the model
                plt.plot(y_list, predicted_vx_3, color = color_list[k], label='Model Prediction')

                y_vals = list(avg_prof.keys()) # getting y-coordinates of the layers
                y_vals = [y_vals[i] - Lx/2 for i in range(len(y_vals))] # shifting y-coordinates of layers by -Lx/2
                # Plotting simulation produced vx vs y
                plt.scatter(y_vals, avg_prof.values(), color=color_list[k], marker = '.', label = f'Npin = {pin_num}')

                
                plt.xlabel(r'$y$')
                plt.ylabel(r'$v_x$', fontsize=20)
                plt.yticks(fontsize=15)
                #plt.ylim(-0.003, 0.003)

             
                plt.legend(loc = 'upper left')
        
                '''You might want to change the next line
                '''
                img_name = f'plot_profile_{gammadot}_inf' # naming saved file
                plt.savefig(img_name)
                


# # Defining functions for creating animations of profiles

# In[ ]:


def animate_profile(num_MDsteps, quantity_str): 
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

    global pin_num
    global gammadot
    global region
    global file_dir
    
    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_MDsteps, interval=1000)
    #frame=Number of profiles/mdsteps
    # interval= how much wait time before showing next frame (in ms)
    # animate is the function defined later on
    
    name = f"{file_dir}/gammadot{gammadot}_pins{pin_num}_{quantity_str}prof_{region}.MP4" 
    #example: [current directory]/1e-06/Pins9/Phi084500/gammadot1e-06_pins9_vxprof_inf.MP4
    
    writervideo = matplotlib.animation.FFMpegWriter(fps=1) # fps = frame per second
    ani.save(name, writer=writervideo) # saving the animation
    plt.show()


# In[ ]:


def animate(n):
    '''Creates n th frame of the animation
    '''
    global Lx 
    global lywall
    global vbottom
    global quantity_str
    global pin_num
    global gammadot
    global profile_dict
    
    # Getting the scale of vx axis 
    min_list = []
    max_list = []
    for layer in profile_dict:
        min_list.append(min(profile_dict[layer][quantity_str]))# appending minimum vx for the layer
        max_list.append(max(profile_dict[layer][quantity_str]))# appending maximum vx for the layer 
    min_quantity = min(min_list) # minimum vx 
    max_quantity = max(max_list) # maximum vx

    # creating lists of y-coordinates and quantities (vx for example) of each layer for n th frame
    tempy = []
    temp_quantity = []
    for i in range(1, 33):
        tempy.append(profile_dict[i]['y'][n-1:n]) 
        temp_quantity.append(profile_dict[i][quantity_str][n-1:n])
        
    plt.cla()#clear the previous graph
    plt.scatter(tempy, temp_quantity)
    plt.xlabel("y-coordinate of layers (y)")
    plt.ylabel(quantity_str)
    plt.ylim(min_quantity, max_quantity)

    ycoord = [lywall, lywall+Lx] # min and max y-coordinates
    affinevelovity = [vbottom, -vbottom] # min and max velocities
    plt.plot(ycoord,affinevelovity) # plotting affine line
    
    str_text = f'$\gamma^.$={gammadot}, Npin={pin_num}'
    plt.text(Lx/4, 0.9*max_quantity, str_text, size = 10)


# # Creating animation of profiles

# In[ ]:


'''Only next few lines you might need to change 
You might also need to change the variables region and file_dir in few lines later
''' 
Ra = 1
Rb = 1.4
Namid = 2048
gammadotbeforepoint_list = [1]
gammadot_list = [6]
pin_list = [9,16,36,64]
file_name = 'profileinf'
quantity_str = 'vx'
lywall = 3*2.8 # thickness of either side of the wall

# Do not change the sequence of codes from here on
for i in range(len(gammadotbeforepoint_list)):
    for j in range(len(gammadot_list)):
        
        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j]))
        
        for k in range(len(pin_list)):
            pin_num = pin_list[k]
            for l in range(len(phi_list)):
                phi_val = phi_list[l]
            
                '''You will encounter region and file_dir soon
                '''
                path_file = dirpath(gammadot, pin_num, phi_val, file_name)
                #example: [current directory]/1e-06/Pins9/Phi084500/profileinf
                # go-to-function: path_file
    
                region = path_file[-3:]
                #example: inf
    
                file_dir = path_file[:-10]
                #example: [current directory]/1e-06/Pins9/Phi084500/
    

                Lx, vbottom, mdstep_list, profile_dict = system_info(gammadot, path_file)
                # go-to-function: system_info
            
                num_mdstep = len(mdstep_list)
                animate_profile(num_mdstep, quantity_str)
                # go-to-function: animate_profile, animate                
                


# ## Velocity Autocorrelation Function Calculation
# 
# $ \large u_y(t) = v_x(y, t) - \left< v_x(y, t) \right>_t$
# 
# $ \large C(y, \tau) = \dfrac{\left< u_y(t) \cdot u_y(t+\tau) \right>_t}{\left< u_y(t)^2 \right>_t} $

# # Defining correlation function
# The next block of code calculates correlation only for the timescales (list of tau) when every tau is a positive integer multiple of first tau. 

# In[ ]:


def corelation(timescales):
    '''Parameters:
    - timescales; list of tau values
    Returns
    A dictionary:
    key: layer IDs
    Value: list of correlation values C(y, tau) for different taus
    '''
    global profile_dict
    global avg_prof
    
    ut_dict = {} # initiating a dictionary to catch u_y(t)
    ctau_dict = {} # initiating a dictionary to catch correlation values
    
    for layer in profile_dict:
        ut_dict[layer] = []
        # this list saves u(t) values for different t for the same layer
        # for example, ut_dict[1] saves u(t) for the 1st layer across all MDsteps in the profile 
            
        ctau_dict[layer] = [] 
        # this list saves C(tau) values for different tau for the same layer
        
      
    for layer in profile_dict:
        vx_list = profile_dict[layer]['vx'] # getting vx values for all t for a layer
        avg_prof_y = list(avg_prof.keys())
        for i in range(len(vx_list)):
        
            ut_dict[layer].append( vx_list[i] - avg_prof[avg_prof_y[layer-1]] ) 
            
    avg_ut_sq = [] # This list saves average u(t)**2 for all the layers
    for layer in ut_dict:
        avg_ut_sq.append( np.mean( np.array(ut_dict[layer])**2 ) )
     
  
    for tau in timescales:
        for layer in profile_dict:
            
            c_tau = [] 
            # This list saves the product in the numerator of C(tau) for a specific tau for a specific layer
            step = int(tau/timescales[0])
            # step determines how much to skip in the list of ut_dict[layer], depending on tau
            # Dividing by timescales[0] as each tau is an integer multiple of the first tau
            for i in range(0, len(ut_dict[layer])-step, step):
                
                c_tau.append( ut_dict[layer][i] * ut_dict[layer][ i+step ] )
            ctau_dict[layer].append( np.mean(c_tau) / avg_ut_sq[layer-1] ) 
            # taking the mean of the product and then dividing by denominator
            
    return ctau_dict          


# # Creating correlation plot

# In[ ]:


# This will help you analyze just for one combination of gammadot, Npin, and phi. 

timescales = [ i for i in range(10000, 2010000, 10000 ) ] 
# list of all tau. Each tau is an integer multiple of the first tau

ctau_dict = corelation( timescales )

for i in range(1,33):
    plt.figure()
    plt.plot(timescales, ctau_dict[i], label= f'Layer = {i}', c = 'r')
    plt.xlabel( r'$\tau$' )
    plt.ylabel( r'$C(\tau)$' )
    plt.legend()

