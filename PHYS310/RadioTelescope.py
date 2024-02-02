#!/usr/bin/env python
# coding: utf-8

# # Radio Telescope - Example of Loading Spectrum Data

# ## Load the Data from a .rad Text File

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


# Necessary Constants 
c = 3*10**8 
f_0 = 1420.406


# ## Determining the line of sight velocity of Orion

# In[ ]:


file="oriondata.rad"                                              # edit in file name
time= np.loadtxt(file, dtype='str', usecols=0, comments='*')     # reads time stamp as a string (for now)
freq0= np.loadtxt(file, dtype='float', usecols=5, comments='*')       # reads the frequency of the first element in the spectrum
freqstep= np.loadtxt(file, dtype='float', usecols=6, comments='*')     # reads the frequency step
nspec = np.loadtxt(file, dtype='int',usecols=8, comments='*')          # reads the number of elements in each spectrum
endspec=9+nspec[0]
mylist2=np.arange(9,endspec,1,dtype=int)            # spectrum is written starting with column 9 and extending for nspec elements
data= np.loadtxt(file, dtype='float', usecols=mylist2, comments='*')   # data is a two dimensional array
vlsr= np.loadtxt(file, dtype='float', usecols=endspec+1, comments='*') # VLSR (velocity of local standard of rest)
freqval = np.linspace(freq0[0],freq0[0]+(nspec[0]-1)*freqstep[0],nspec[0])


# In[ ]:


avgspec = np.mean(data,axis=0)  # average all the rows together in the data array to reduce noise
plt.figure()
plt.ylim((370,480))          # setting up y-limits to make the plot look nice; not necessary
plt.plot(freqval,avgspec)
plt.title("Radio Telescope spectrum")
plt.xlabel('Frequency (MHz)')
plt.ylabel("Intensity (arbitrary units)")


# In[ ]:


f_0 = 1420.406
f_obs = 1420.183
c = 3*10**8
v_obs = (f_0 - f_obs)*c/(f_0*1000)
print('Observed velocity =', v_obs, 'km/s' )
avg_vlsr = np.mean(vlsr)
v_los = v_obs - avg_vlsr
print('Line of Sight velocity =', v_los, 'km/s' )


# ## Analyzing First data set (Apr 19) of Longitudes ranging from 10-90 degrees

# In[ ]:


vlsr_list = [] # Saving average vlsr value from each longitude data
for i in range(10,100, 10):
    file=f"data19/april19_{i}_test.rad"                                              # edit in file name
    time= np.loadtxt(file, dtype='str', usecols=0, comments='*')     # reads time stamp as a string (for now)
    freq0= np.loadtxt(file, dtype='float', usecols=5, comments='*')       # reads the frequency of the first element in the spectrum
    freqstep= np.loadtxt(file, dtype='float', usecols=6, comments='*')     # reads the frequency step
    nspec = np.loadtxt(file, dtype='int',usecols=8, comments='*')          # reads the number of elements in each spectrum
    endspec=9+nspec[0]
    mylist2=np.arange(9,endspec,1,dtype=int)            # spectrum is written starting with column 9 and extending for nspec elements
    data= np.loadtxt(file, dtype='float', usecols=mylist2, comments='*')   # data is a two dimensional array
    vlsr= np.loadtxt(file, dtype='float', usecols=endspec+1, comments='*') # VLSR (velocity of local standard of rest)
    vlsr_list.append(np.mean(vlsr))
    freqval = np.linspace(freq0[0],freq0[0]+(nspec[0]-1)*freqstep[0],nspec[0]) 
    avgspec = np.mean(data,axis=0)  # average all the rows together in the data array (why?)

    plt.figure()
    # plt.ylim((1500,2300))          # setting up y-limits to make the plot look nice; not necessary
    plt.ylim(6350, 9700)
    plt.plot(freqval,avgspec,label=f'$l={i}^o$' )

    plt.title("Radio Telescope spectrum")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel("Intensity (arbitrary units)")
    plt.legend()
    plt.savefig(f'data19/Apr19_l{i}.png')


# ## Using scipy Filter to get rid of some noise

# In[ ]:


from scipy.signal import lfilter

n = 20
# the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b, a, avgspec)
plt.figure()
plt.ylim(6000, 9000)
plt.plot(freqval, yy, linewidth=2, linestyle="-", c="b")
plt.title("Radio Telescope spectrum")
plt.xlabel('Frequency (MHz)')
plt.ylabel("Intensity (arbitrary units)")


# ## Constructing vrot vs r

# In[ ]:


# After and before represents the highest and lowest initial frequencies
#and associated veocity calculations

freq_list_after = [1420.243, 1420.218, 1420.218, 1420.207, 1420.216, 1419.953, 1420.018, 1420.174, 1420.221, 1420.275, 1420.394, 1420.377, 1420.400 ]
freq_list_before = [1420.219, 1420.090, 1420.103,1420.100, 1420.010, 1419.853, 1419.953, 1420.150, 1420.205, 1420.222, 1420.326, 1420.330, 1420.353 ]

v_obs_list_before = [(f_0 - f_obs)*c/(f_0*1000) for f_obs in freq_list_before]
v_obs_list_after = [(f_0 - f_obs)*c/(f_0*1000) for f_obs in freq_list_after]

v_los_list_before = [v_obs_list_before[i] - vlsr_list[i] for i in range(len(freq_list_before))]
v_los_list_after = [v_obs_list_after[i] - vlsr_list[i] for i in range(len(freq_list_after))]


# In[ ]:


vrot_before = [v_los_list_before[i] + 230*np.sin(10*(i+1)*np.pi/180) for i in range(len(v_los_list_before))]
vrot_after = [v_los_list_after[i] + 230*np.sin(10*(i+1)*np.pi/180) for i in range(len(v_los_list_after))]


# In[ ]:


longitude = [10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90 ]
distance = [27070.979*365*24*3.6*3*10**8*np.sin(longitude[i]*np.pi/180) for i in range(len(longitude))]


# In[ ]:


plt.figure()
plt.scatter(distance, vrot_before, marker='x', c='b', label=r'Earliest $f_{obs}$')
plt.scatter(distance, vrot_after, marker='x', c='r', label=r'Latest $f_{obs}$')
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'$v_{rot}(r)$ (km/s)')
plt.legend(loc=4)


# ## Testing Noise Level on Thurs Apr 20

# In[ ]:


file="testnoise.rad"                                              # edit in file name
time= np.loadtxt(file, dtype='str', usecols=0, comments='*')     # reads time stamp as a string (for now)
freq0= np.loadtxt(file, dtype='float', usecols=5, comments='*')       # reads the frequency of the first element in the spectrum
freqstep= np.loadtxt(file, dtype='float', usecols=6, comments='*')     # reads the frequency step
nspec = np.loadtxt(file, dtype='int',usecols=8, comments='*')          # reads the number of elements in each spectrum
endspec=9+nspec[0]
mylist2=np.arange(9,endspec,1,dtype=int)            # spectrum is written starting with column 9 and extending for nspec elements
data= np.loadtxt(file, dtype='float', usecols=mylist2, comments='*')   # data is a two dimensional array
vlsr= np.loadtxt(file, dtype='float', usecols=endspec+1, comments='*') # VLSR (velocity of local standard of rest)
freqval = np.linspace(freq0[0],freq0[0]+(nspec[0]-1)*freqstep[0],nspec[0])


# In[ ]:


avgspec = np.mean(data,axis=0)  # average all the rows together in the data array to reduce noise
plt.figure()
#plt.ylim((370,480))          # setting up y-limits to make the plot look nice; not necessary
plt.plot(freqval,avgspec)
plt.title("Radio Telescope spectrum")
plt.xlabel('Frequency (MHz)')
plt.ylabel("Intensity (arbitrary units)")


# ## Combining data of April 19,20,21,22,23

# In[ ]:


# Changing filenames to access them

cwd = os.getcwd() 

for i in range(19,24):
    rel_dir = f'data{i}'
    dir_data = os.path.join(cwd, rel_dir)
    all_files = os.listdir(dir_data)
    for file in all_files:

        file_name, file_extension = os.path.splitext(f'{dir_data}/{file}')

        if file_extension == '.rad':
            new_filename = 'l' + file[8:]
            file_path = os.path.join(dir_data, file)
            os.replace( file_path, f'alldata/{new_filename}' )
            


# In[4]:


cwd = os.getcwd()
data_dir = os.path.join(cwd, 'alldata')
os.chdir(data_dir)
all_files = os.listdir(os.getcwd())
all_files.sort(key = lambda x: x[1:3])
long_list = [] # initiating a list to capture longitudes

vlsr_list = [] # Saving average vlsr value from each longitude data

for file in all_files:
    
    time= np.loadtxt(file, dtype='str', usecols=0, comments='*')     # reads time stamp as a string (for now)
    freq0= np.loadtxt(file, dtype='float', usecols=5, comments='*')       # reads the frequency of the first element in the spectrum
    freqstep= np.loadtxt(file, dtype='float', usecols=6, comments='*')     # reads the frequency step
    nspec = np.loadtxt(file, dtype='int',usecols=8, comments='*')          # reads the number of elements in each spectrum
    endspec=9+nspec[0]
    mylist2=np.arange(9,endspec,1,dtype=int)            # spectrum is written starting with column 9 and extending for nspec elements
    data= np.loadtxt(file, dtype='float', usecols=mylist2, comments='*')   # data is a two dimensional array
    vlsr= np.loadtxt(file, dtype='float', usecols=endspec+1, comments='*') # VLSR (velocity of local standard of rest)
    
    vlsr_list.append(np.mean(vlsr))
    freqval = np.linspace(freq0[0],freq0[0]+(nspec[0]-1)*freqstep[0],nspec[0]) 
    avgspec = np.mean(data,axis=0)  # average all the rows together in the data array (why?)

    plt.figure()
    # plt.ylim((1500,2300))          # setting up y-limits to make the plot look nice; not necessary
    # plt.ylim(6350, 9700)
    l = file[1:3]
    long_list.append(int(l))
    plt.plot(freqval,avgspec,label=f'$l={l}^o$' )

    plt.title("Radio Telescope spectrum")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel("Intensity (arbitrary units)")
    plt.legend()
    # plt.savefig(f'data19/Apr19_l{i}.png')


# ## Analyzing each day's data separately

# In[ ]:


vlsr_list = [] # Saving average vlsr value from each longitude data
# changing the date and range each time for new dataset
for i in range(10,22, 2):
    file=f"data20/april20_{i}_test.rad"                                              # edit in file name
    time= np.loadtxt(file, dtype='str', usecols=0, comments='*')     # reads time stamp as a string (for now)
    freq0= np.loadtxt(file, dtype='float', usecols=5, comments='*')       # reads the frequency of the first element in the spectrum
    freqstep= np.loadtxt(file, dtype='float', usecols=6, comments='*')     # reads the frequency step
    nspec = np.loadtxt(file, dtype='int',usecols=8, comments='*')          # reads the number of elements in each spectrum
    endspec=9+nspec[0]
    mylist2=np.arange(9,endspec,1,dtype=int)            # spectrum is written starting with column 9 and extending for nspec elements
    data= np.loadtxt(file, dtype='float', usecols=mylist2, comments='*')   # data is a two dimensional array
    vlsr= np.loadtxt(file, dtype='float', usecols=endspec+1, comments='*') # VLSR (velocity of local standard of rest)
    vlsr_list.append(np.mean(vlsr))
    freqval = np.linspace(freq0[0],freq0[0]+(nspec[0]-1)*freqstep[0],nspec[0]) 
    avgspec = np.mean(data,axis=0)  # average all the rows together in the data array (why?)

#     plt.figure()
    # plt.ylim((1500,2300))          # setting up y-limits to make the plot look nice; not necessary
    #plt.ylim(6350, 9700)
    plt.plot(freqval,avgspec,label=f'$l={i}^o$' )

    plt.title("Radio Telescope spectrum")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel("Intensity (arbitrary units)")
    plt.legend()
    #plt.savefig(f'data19/Apr19_l{i}.png')


# In[8]:


# The dictionary containing longitudes as keys and f_obs as values

data={2:[1420.315], 5:[1420.294], 7:[1420.291], 10:[1420.222,1420.225], 12:[1420.223], 14:[1420.141], 15:[1419.942], 16:[1420.122], 18:[1420.014], 20:[1419.852,1419.988], 22:[1419.918], 25:[1419.968], 27:[1419.988], 30:[1419.975], 32:[1420.062], 35:[1420.048], 37:[1420.106], 40:[1420.145], 42:[1420.145], 45:[1420.162], 47:[1420.195], 50:[1420.204], 52:[1420.210], 55:[1420.249], 57:[1420.259], 60:[1420.306], 62:[1420.325], 65:[1420.334], 67:[1420.369], 70:[1420.338], 72:[1420.360], 75:[1420.368], 77:[1420.386], 80:[1420.367], 82:[1420.312], 85:[1420.387], 87:[1420.377], 90:[1420.365]}


# In[9]:


# For some longitudes, multiple observations have been made; taking mean in that case

for key in data:
    if len(data[key])>1:
        data[key]= np.mean(data[key])
    else:
        data[key] = data[key][0]


# In[10]:


# Sorting by longitude value
longitudes = list(data.keys())
longitudes.sort()
sorted_data = {i: data[i] for i in longitudes}


# In[22]:


# Calculating necessary quantities

v_obs = [(f_0 - f_obs)*c/(f_0*1000) for f_obs in sorted_data.values()]

v_los_list = [v_obs[i] - vlsr_list[i] for i in range(len(list(sorted_data.values())))]

vrot = [v_los_list[i] + 230*np.sin(longitudes[i]*np.pi/180) for i in range(len(v_los_list))]

distance = [ 8.2782376*np.sin(longitudes[i]*np.pi/180) for i in range(len(longitudes)) ]

Mass_encl = [vrot[i]**2/(6.673*10**(-11))*(distance[i]*1000*3.086e16)*10**6 for i in range(len(distance))]

density_encl = [vrot[i]**2/(4*np.pi*6.673*10**(-11)*(distance[i]*1000*3.086e16)**2)*10**6 for i in range(len(distance))]


# In[ ]:


# Plotting vrot vs r
plt.figure()
plt.scatter(distance, vrot, marker='x', c='b')
plt.xlabel(r'$r$ (kPc)')
plt.ylabel(r'$v_{rot}$ ($kms^{-1}$)')
plt.ylim(0,300)
plt.title('Galaxy Rotation Curve')


# In[ ]:


# Plotting enclosed M vs r
Mass_encl_solar = [ Mass_encl[i]*(1/(2*10**30)) for i in range(len(Mass_encl)) ]
plt.figure()
plt.scatter(distance, Mass_encl_solar, marker='x', c='b')
plt.yscale('log')
plt.xlabel(r'$r$ (kPc)')
plt.ylabel(r'M ($M_0$)')
plt.title('Mass Curve')


# In[20]:


# converting density unit solar mass per pc^3
density_encl_solar = [ density_encl[i]/(2*10**30)*(3.086*10**16)**3 for i in range(len(density_encl))]


# In[ ]:


# Plotting enclosed density vs r
plt.figure()
plt.scatter(distance, density_encl_solar, marker='.', c='b')
plt.xlabel(r'  $r$ (kPc)')
plt.ylabel(r' $\rho$ ($\frac{M_0}{Pc^3}$)')
plt.title('Galaxy Density Curve')


# In[16]:


# Plotting vrot vs r
plt.figure()
plt.scatter(distance[11:], vrot[11:], marker='x', c='b')
plt.xlabel(r'$r$ (kPc)')
plt.ylabel(r'$v_{rot}$ ($kms^{-1}$)')
# plt.ylim(0,300)
plt.title('Galaxy Rotation Curve')


# In[18]:


# Plotting enclosed M vs r
Mass_encl_solar = [ Mass_encl[i]*(1/(2*10**30)) for i in range(len(Mass_encl)) ]
plt.figure()
plt.scatter(distance[11:], Mass_encl_solar[11:], marker='x', c='b')
# plt.yscale('log')
plt.xlabel(r'$r$ (kPc)')
plt.ylabel(r'M ($M_0$)')
plt.title('Mass Curve')


# In[21]:


# Plotting enclosed density vs r
plt.figure()
plt.scatter(distance[11:], density_encl_solar[11:], marker='.', c='b')
plt.xlabel(r'  $r$ (kPc)')
plt.ylabel(r' $\rho$ ($\frac{M_0}{Pc^3}$)')
plt.title('Galaxy Density Curve')


# In[24]:


# Plotting vrot vs r
plt.figure()
plt.scatter(distance, v_los_list, marker='x', c='b')
plt.xlabel(r'$r$ (kPc)')
plt.ylabel(r'$v_{tan}$ ($kms^{-1}$)')
plt.ylim(0,300)
plt.title('Galaxy Rotation Curve')


# In[ ]:




