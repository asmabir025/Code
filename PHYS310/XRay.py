#!/usr/bin/env python
# coding: utf-8

# In[81]:


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
plt.style.use('seaborn-whitegrid')
import matplotlib.patches as patches


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


delta_t_list = [i for i in range(10,90,10)] + [100, 150, 200, 250, 500] # creating a list with delta_t values from the experiments


# In[4]:


steep_bmin_list = [3.7, 4.0,3.9, 3.8, 3.8, 3.9, 4.0, 4.0, 4.1, 3.9, 3.9, 3.8, 3.9] # minimum angles for steepest slope
shallow_bmin_list = [3.4, 3.6, 3.4, 3.6, 3.4, 3.5, 3.5, 3.5, 3.4, 3.7, 3.6, 3.6, 3.5] #minimum angles for shallowest slope


# In[5]:


# Just wanted to check how our data looks like

plt.figure()
plt.xlabel('$\delta_{t}$ (s)')
plt.ylabel('Minimum Angle')
plt.plot(delta_t_list, steep_bmin_list, 'ro', label = 'steep angle values')
plt.plot(delta_t_list, shallow_bmin_list, 'og', label = 'shallow angle values')
plt.legend()


# In[6]:


d = 282.01*10**(-12) # distance between two planes of NaCl crystal
echarge = 1.6*10**(-19) 
U = 25000 # voltage
c = 3*10**8 #light speed
steep_bmin_rad = [i*np.pi/180 for i in steep_bmin_list] # 'steep' angles in radian
shallow_bmin_rad = [i*np.pi/180 for i in shallow_bmin_list] # shallow angles in radian
# print(steep_bmin_rad)


# In[7]:


#creating minimum wavelength list for both steep and shallow slopes
lambda_steeplist = [2*d*(np.sin(i)) for i in steep_bmin_rad]
lambda_shallowlist = [2*d*(np.sin(i)) for i in shallow_bmin_rad]


# In[8]:


# calculating Plank Constant
h_max_list = [2*d*echarge*U*(np.sin(i))/c for i in steep_bmin_rad]
h_min_list = [2*d*echarge*U*(np.sin(i))/c for i in shallow_bmin_rad]


# In[9]:


# creating horizontal line for two means
mean_hmax = np.mean(h_max_list)
mean_hmin = np.mean(h_min_list)
hmax_line = []
hmin_line = []
delta_t = []
for i in range(0, max(delta_t_list)):
    delta_t.append(i)
    hmax_line.append(mean_hmax)
    hmin_line.append(mean_hmin)


# In[10]:


plt.figure()
plt.xlabel('$\Delta t (s)$')
plt.ylabel('Plank Constant h (Js)')
plt.plot(delta_t_list, h_max_list, 'ro', label = 'steep angle values')
plt.plot(delta_t_list, h_min_list, 'og', label = 'shallow angle values')
plt.plot(delta_t, hmax_line, 'r')
plt.plot(delta_t, hmin_line, 'g')
plt.legend()
# plt.savefig('h_calculation')


# ###Analyzing Tom's Data to calculate Plank's Constant
# 
# Anode: Mo
# 
# $Δt = 50s$
# 
# $\Delta β = 0.1$ (Degree)
# 
# $β = 2.5 - 30$ (Degree)
# 
# I = 1.0 mA
# 
# U = 20, 25, 30, 35 kV

# In[11]:


angle, r_20, r_25, r_30, r_35 = np.loadtxt("file_1.dat.txt", skiprows=1, usecols = (0,1,2,3,4), dtype='float', unpack=True)
# list of angles, rates for different U = 20, 25, 30, 35


# In[12]:


plt.figure()
plt.xlabel('$Angle^{\degree}$')
plt.ylabel('Count Rate ($s^{-1}$)')
plt.plot(angle, r_20, color ='red', label = 'U = 20kV, Mo')
plt.plot(angle, r_25, color ='orange', label = 'U = 25kV, Mo')
plt.plot(angle, r_30, color ='green', label = 'U = 30kV, Mo')
plt.plot(angle, r_35, color ='blue', label = 'U = 35kV, Mo')
plt.legend()
plt.savefig('Mo_U.png')


# In[13]:


rlist = [r_20, r_25, r_30, r_35] # rate lists of linear region for different U's
for i in range(4):
  plt.figure()
  plt.plot(angle[:75], rlist[i][:75], '.') #plotting to find the linear range of the plot; eye-estimated this region from the above


# In[23]:


period = 3 # number of points to determine slope of a point
start_indexlist = [36, 16, 23, 10] # start indices of linear region for different U's
end_indexlist = [42, 23, 30, 17]  # end indices of linear region for different U's
rlist = [r_20, r_25, r_30, r_35] # rate lists of linear region for different U's
d = 282.01*10**(-12) 
echarge = 1.6*10**(-19)
U_list = [20000, 25000, 30000, 35000] # creating voltage list
c = 3*10**8
h_max_T = [] #iniitating a list to put h values of maximum slopes for different U's
h_min_T = [] #iniitating a list to put h values of minimum slopes for different U's


# In[24]:


for i in range(4): # 4 because 4 different U
    start_index = start_indexlist[i]
    end_index = end_indexlist[i]
    r = rlist[i]

    slope_list = [] # initiating list for slopes of the points in linear region
    for ang in range(start_index, end_index): #angles through linear region
        angle_period = [] # initiating list for saving 4 angles beside the specific angle
        r_period = [] # initiating list for saving 4 count rates beside the specific angle
        for k in range(ang - period//2,  ang + period//2 + 1): # looking into two points left and two points and the specific point
            angle_period.append(angle[k])
            r_period.append(r[k])
        slope_list.append(np.polyfit(angle_period, r_period, 1)[0]) # getting slope for linear regression
    #print(slope_list)

    max_slope = max(slope_list)
    min_slope = min(slope_list)
    max_ang_index = start_index + np.where(slope_list == max_slope)[0][0] # getting minimum angle for maximum slope
    min_ang_index = start_index + np.where(slope_list == min_slope)[0][0] # getting minimum angle for minimum slope

    angle_max = angle[max_ang_index]*np.pi/180
    angle_min = angle[min_ang_index]*np.pi/180

    voltage = U_list[i]
    h_maxt = 2*d*echarge*voltage*(np.sin(angle_max))/c
    h_mint = 2*d*echarge*voltage*(np.sin(angle_min))/c
    h_min_T.append(h_mint) 
    h_max_T.append(h_maxt)


# In[25]:


def std_dev(list1, meann):
  '''
  Parameter : list1, a list of float values
              meann, a float value
  Given a list and its mean, the function returns standard deviation
  '''

  std_sum = 0

  for i in range(len(list1)):
    std_sum += (list1[i] - meann)**2

  return (std_sum/(len(list1)-1))**0.5


# In[17]:


avg_h_list = [] # initiating a list to catch average h for each point from its 'shallow' and 'steep' values
uncert_h_list = []
for i in range(len(h_min_list)):
  avg_h_list.append( (h_max_list[i] + h_min_list[i])/2 ) # determining h for each measurement on Cu anode
  uncert_h_list.append( abs(h_max_list[i] - h_min_list[i])/2 ) # determining uncertainty for each measurement on Cu anode

for i in range(len(h_min_T)):
  avg_h_list.append( (h_max_T[i] + h_min_T[i])/2 ) # determining h for each measurement on Mo anode (Tom's data)
  uncert_h_list.append( abs(h_max_T[i] - h_min_T[i])/2 ) # determining uncertainty for each measurement on Mo anode (Tom's data)


# In[27]:


# calculating weighted average and uncertainty of h measurement including both Tom's and ours: just copied and pasted Jackie's code and modified variable names

hvalues = np.array(avg_h_list)
uncert = np.array(uncert_h_list)

weights = 1.0/uncert**2
hWeightedAvg = np.dot(hvalues,weights)/np.sum(weights)
αh = np.sqrt(1.0/np.sum(weights))
hWeightedAvg, αh


# In[28]:


# weighted calculation on just Tom's data

hvaluesT = np.array(avg_h_list[-4:]) # only last 4 data points of the list are from Tom's data
uncertT = np.array(uncert_h_list[-4:])

weightsT = 1.0/uncertT**2
hWeightedAvgT = np.dot(hvaluesT,weightsT)/np.sum(weightsT)
αhT = np.sqrt(1.0/np.sum(weightsT))
hWeightedAvgT, αhT


# In[29]:


# weighted calculation on just our data

hvaluesCu = np.array(avg_h_list[:-4]) # all points except last 4 are from our data
uncertCu = np.array(uncert_h_list[:-4])

weightsCu = 1.0/uncertCu**2
hWeightedAvgCu = np.dot(hvaluesCu,weightsCu)/np.sum(weightsCu)
αhCu = np.sqrt(1.0/np.sum(weightsCu))
hWeightedAvgCu, αhCu


# In[104]:


num_T = [i+1 for i in range(len(hvaluesT))] #creating a list for number of measurements for Tom's data on Molybdenum
num_Cu = [i+1 for i in range(len(hvaluesCu))] #creating a list for number of measurements for our data on Copper

# plt.figure()
fig, ax = plt.subplots()
ax.text(5.5, 6.75*10**(-34), r'Mo Data', fontsize=22)
ax.text(10, 5.5*10**(-34), r'Cu Data', fontsize=22, color = 'blue')
ax.set_xticks([])
ax.set_ylabel(r"Planck's Constant, h (Js)", fontsize = 17)
ax.set_xlabel(r"Measurements", fontsize = 17)

rect1 = patches.Rectangle((0.5, 5.5*10**(-34)), 3.9, 3*10**(-34), linewidth=5, edgecolor='black', facecolor='none')
rect2 = patches.Rectangle((0.5, 4.4*10**(-34)), 13, 1*10**(-34), linewidth=5, edgecolor='blue', facecolor='none')
ax.errorbar(num_T, hvaluesT, yerr=uncertT, fmt='o',ecolor='gray', elinewidth=3, capsize=0)
ax.errorbar(num_Cu, hvaluesCu, yerr=uncertCu, fmt='o',ecolor='lightsteelblue', elinewidth=3, capsize=0)
    
ax.errorbar(5, hWeightedAvgT, yerr = αhT, fmt='o',ecolor='black', elinewidth=3, capsize=0, label = 'Average h (Mo)')
ax.errorbar(14, hWeightedAvgCu, yerr = αhCu, fmt='o',ecolor='blue', elinewidth=3, capsize=0, label = 'Average h (Cu)')
ax.add_patch(rect1)
ax.add_patch(rect2)
plt.legend()
plt.savefig('error_estimate.png')


# In[ ]:




