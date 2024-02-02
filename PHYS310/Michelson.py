#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import matplotlib  as mpl
import matplotlib.pyplot as plt
from scipy import optimize
import statistics as stats
# Following is an Ipython magic command that puts figures in notebook.
get_ipython().run_line_magic('matplotlib', 'notebook')
        
# M.L. modifications of matplotlib defaults
# Changes can also be put in matplotlibrc file, 
# or effected using mpl.rcParams[]
mpl.style.use('classic') 
plt.rc('figure', figsize = (6, 4.5)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot params for new size


# In[60]:


import statistics as stats


# # Feb 28 Data

# In[61]:


delta_n = [10, 15, 20, 25, 30 ]
# For each delta_n, we took three data points of delta_d. Three points are minimum to get std.
delta_d = [[3.1,3,2.9],[4.5, 4.4, 4.4],[5.9,5.7,5.8],[7.2,7.1,7], [8.8,8.7,8.5] ] 


# In[63]:


# Getting avergae delta_d and standard deviation for each delta_n case
avg_delta_d = []
std_delta_d = []


for i in range(len(delta_d)):
    avg_delta_d.append(stats.mean(delta_d[i]))
    std_delta_d.append(stats.stdev(delta_d[i])/np.sqrt(len(delta_d[i]))) #standard error; stats module takes account of n-1 in std                 
    


# In[67]:


# This is a discontinuity to our previous plan, just checking how each delta_d/delta_n should be without averaging and stuffs.
#In a sense, this is still raw data

# getting the ratios of delta_d/delta_n for each data point
# delta_d/delta_n should be constant, so making one list (no nesting)

delta_d_notnested = []
for i in range(len(delta_d)):
    
    for j in range(len(delta_d[i])):
        
        delta_d_notnested.append(delta_d[i][j]/delta_n[i])
list1 = [i for i in range(len(delta_d_notnested))]
plt.scatter(list1, delta_d_notnested, label = 'Not tested')
plt.xlabel(f'$\Delta n$')
plt.ylabel(f'$\Delta d / \Delta n$')
plt.legend()


# In[68]:


# this is the averaged data

plt.figure()
plt.errorbar(delta_n, avg_delta_d, std_delta_d, fmt='ok')
plt.xlabel(r'$\Delta$n')
plt.ylabel(r'$\Delta$d')
plt.xlim(5,35)
plt.savefig('delta_d_vs_delta_n.png')


# In[73]:


# calculating delta d/ delta n from average delta_d
frac_d_n = [avg_delta_d[i]/delta_n[i] for i in range(len(delta_n))]
avg_frac = np.mean(np.array(frac_d_n)) # average delta_d/delta_n for all data points

# calculating error of delta d/ delta n for each delta_n
frac_er = [std_delta_d[i]/delta_n[i] for i in range(len(delta_n))] 


# In[72]:


plt.figure()
plt.errorbar(delta_n, frac_d_n, frac_er, fmt='ok')
plt.xlabel(r'$\Delta$n')
plt.ylabel(r'$\Delta d / \Delta n$')
plt.axhline(avg_frac)
plt.xlim(5,35)
plt.savefig('delta_d_frac_vs_delta_n.png')


# In[74]:


# calculating wavelength for all delta_d/delta_n
lambdaM = [2000*frac_d_n[i] for i in range(len(delta_n))] # smallest unit in the knob: 0.1 micron, and n*lambda = 2d
delta_lambda = [2000*frac_er[i] for i in range(len(delta_n))] # uncertainty in lambda


# In[75]:


plt.figure()
plt.errorbar(delta_n, lambdaM, delta_lambda, fmt='ok')
plt.xlabel(r'$\Delta$n')
plt.ylabel(r'$\lambda$')
plt.xlim(5,35)
plt.savefig('lambda_vs_delta_n.png')


# ## March 1
# $\Delta_n = 20$
# 20 Revolutions (equivalent to $\Delta_n = 100$) at first

# In[76]:


# Those are the readings from the knob from our continuous data taking 
read_list_1 = [0, 5.7, 1.3, 6.9, 2.4, 7.9, 3.3,9.2, 4.6, 0.3, 6.3, 2, 8.1, 3.5]
read_list_2 = [8.7, 4.1, 9.6, 5.0, 0.5, 6.0, 1.2, 6.6, 2.1, 7.6, 2.7, 8.15, 3.6, 8.8]


# In[77]:


# calibrated considering every turn brings the knob 1 to 10 and back to 1; for the first list, Cameron read the knob, and for the second, I read it

d_cameron_list = [5.7,11.3, 16.9, 22.4, 27.9, 33.3, 39.2, 44.6, 50.3, 56.3, 62, 68.1,73.5]
d_abir_list = [78.7, 84.1, 89.6, 95, 100.5, 106, 111.2, 116.6, 122.1, 127.6, 132.7, 138.15, 143.6, 148.8]
    


# In[78]:


n_list_cameron = [20*i for i in range(len(d_cameron_list))]
n_list_abir = [260+ 20*i for i in range(len(d_abir_list))] # Cameron stopped at n = 240


# In[79]:


plt.figure()
plt.xlabel('n')
plt.ylabel('d')
plt.scatter(n_list_cameron, d_cameron_list, label='Cameron')
plt.legend(loc = 'upper left')


# In[80]:


plt.figure()
plt.xlabel('n')
plt.ylabel('d')
plt.scatter(n_list_abir, d_abir_list, label='Abir')
plt.legend(loc = 'upper left')


# In[85]:


plt.figure()
plt.scatter(n_list_cameron + n_list_abir, d_cameron_list+d_abir_list, label='Combined')
plt.xlabel('n')
plt.ylabel('d')
plt.legend(loc = 'upper left')


# In[86]:


slope_int_both, cov_both = np.polyfit(n_list_abir+n_list_cameron, d_abir_list+d_cameron_list, 1, cov=True)
fit_list_both = [n*slope_int_both[0] + slope_int_both[1] for n in n_list_abir+n_list_cameron]
residual_both = [fit_list_both[i] - (d_abir_list+d_cameron_list)[i] for i in range(len(n_list_abir+n_list_cameron))]
plt.figure()
plt.scatter(n_list_abir+n_list_cameron, d_abir_list+d_cameron_list, label = 'Combined')
plt.plot(n_list_abir+n_list_cameron, fit_list_both)
plt.xlabel('n')
plt.ylabel('d')
plt.legend(loc = 'upper left')


# In[87]:


plt.figure()
plt.scatter(n_list_abir+n_list_cameron, residual_both, label = 'Combined')
plt.xlabel('n')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')


# It seems like the residual plot has pattern. It might be that hysterisis still affected the data despite the fact that we turned almost $\Delta n = 120$ before even starting the data. Or, it could be that Cameron and switched our roles at the middle, but that happened around $\Delta n = 250$
# 
# Next step: Separate analysis of our data

# In[91]:


#linear fitting only for Cameron's data
slope_int_cam, cov_cam = np.polyfit(n_list_cameron, d_cameron_list, 1, cov=True) 


# In[92]:


fit_list_cameron = [n*slope_int_cam[0] + slope_int_cam[1] for n in n_list_cameron]


# In[93]:


residual = [fit_list_cameron[i] - d_cameron_list[i] for i in range(len(n_list_cameron))]


# In[94]:


plt.figure()
plt.scatter(n_list_cameron, d_cameron_list, label = 'Cameron')
plt.plot(n_list_cameron, fit_list_cameron, color = 'red')
plt.xlabel('n')
plt.ylabel('d')
plt.legend(loc = 'upper left')


# In[95]:


plt.figure()
plt.scatter(n_list_cameron, residual, label = 'Cameron')
plt.xlabel('n')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')


# There seems to have patterns there; Jackie pointed out that the residual values are high compared to those from combined analysis.

# In[96]:


# linear fitting for my data

slope_int_ab, cov_ab = np.polyfit(n_list_abir, d_abir_list, 1, cov=True)
fit_list_abir = [n*slope_int_ab[0] + slope_int_ab[1] for n in n_list_abir]
residual_abir = [fit_list_abir[i] - d_abir_list[i] for i in range(len(n_list_abir))]
plt.figure()
plt.scatter(n_list_abir, d_abir_list, label = 'Abir')
plt.plot(n_list_abir, fit_list_abir)
plt.xlabel('n')
plt.ylabel('d')
plt.legend(loc = 'upper left')


# In[97]:


plt.figure()
plt.scatter(n_list_abir, residual_abir, label='Abir')
plt.xlabel('n')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')


# There is a pattern in every three points, I don't know. Later (March 29) Jackie pointed out that the values of residuals are low. This might mean that we probably did not turn enough during Cameron's collection of data to avoid the hysterisis. 

# In[98]:


# Wavelength from Cameron in nm
2000*slope_int_cam[0]


# In[99]:


# Wavelength from me in nm
2000*slope_int_ab[0]


# In[102]:


# I was just checking if any part of my data set produces wavelength closer to the expected value (546nm)
# I did this at Mar 29

for i in range(3, len(n_list_abir)):
    slope_int, cov_val = np.polyfit(n_list_abir[:i],d_abir_list[:i], 1, cov = True )
    print('Hg Wavelength is', 2000*slope_int[0], '+/-', 2000*np.sqrt(cov_val[0][0]), 'nm')


# This seems that my data points from i = 0 (n = 260) to i = 5 (n = 360) had the average wavelength of expected value with a low error (compared to my other analyis for higher and lower i's). As we were taking data for a long time, the last few values might be associated with higher uncertainties. 

# In[106]:


# Just out of curiosity, I am going to calculate residual for that fit with the expected value above. 
# I did this at Mar 29

slope_int_ab, cov_ab = np.polyfit(n_list_abir[:6], d_abir_list[:6], 1, cov=True)
fit_list_abir = [n*slope_int_ab[0] + slope_int_ab[1] for n in n_list_abir[:6]]
residual_abir = [fit_list_abir[i] - d_abir_list[i] for i in range(6)]
plt.figure()
plt.scatter(n_list_abir[:6], residual_abir, label='Abir')
plt.xlabel('n')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')


# 6 data points, higher residual, probably nothing new to look into here!

# ## March 28
# Na Wavelength Measurement

# In[35]:


reading_list = [4.2, 8.7, 3.2, 7.7, 2.2, 6.6, 1.1, 5.6, 0, 4.7, 9.2, 3.7, 8.1, 2.8, 7.25, 1.7, 6.1, 0.5, 4.9, 9.4, 3.8, 8.2, 2.6, 7.1, 1.55, 6, 0.4, 4.8, 9.2, 3.6, 8.1, 2.5, 6.95, 1.3, 5.75  ] 


# In[107]:


# After calibration of reading list
d_list_Na = [0, 4.2, 8.7, 13.2, 17.7, 22.2, 26.6, 31.1, 35.6, 40, 44.7, 49.2, 53.7, 58.1, 62.8, 67.25, 71.7, 76.1, 80.5, 84.9, 89.4, 93.8, 98.2, 102.6, 107.1, 111.55, 116, 120.4, 124.8, 129.2, 133.6, 138.1, 142.5, 146.95, 151.3, 155.75]


# In[108]:


n_list_Na = [15*i for i in range(len(d_list_Na))]


# In[109]:


plt.figure()
plt.xlabel('n')
plt.ylabel('d')
plt.scatter(n_list_Na, d_list_Na, label='Na')
plt.legend(loc = 'upper left')


# In[112]:


#linear fitting
slope_int_Na, cov_Na = np.polyfit(n_list_Na, d_list_Na, 1, cov=True)
fit_list_Na = [n*slope_int_Na[0] + slope_int_Na[1] for n in n_list_Na]
residual_Na = [fit_list_Na[i] - d_list_Na[i] for i in range(len(n_list_Na))]
plt.figure()
plt.scatter(n_list_Na, d_list_Na, label = 'Na')
plt.plot(n_list_Na, fit_list_Na)
plt.xlabel('n')
plt.ylabel('d')
plt.legend(loc = 'upper left')


# In[111]:


plt.figure()
plt.scatter(n_list_Na, residual_Na, label='Na')
plt.xlabel('n')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')


# Similar pattern as before from combined data of Cameron and mine. The residuals are higher though.

# In[113]:


# Na wavelenght in nm
2000*slope_int_Na[0]


# In[167]:


# I was just checking if any part of my data set produces wavelength closer to the expected value (546nm)
# I ended up using the data set, containing the later half of the total data
for i in range(len(n_list_Na)//2, len(n_list_Na)//2+10):
    
    slope_int, cov_val = np.polyfit(n_list_Na[i:],d_list_Na[i:], 1, cov = True )
    fit_list_Na = [n*slope_int[0] + slope_int[1] for n in n_list_Na[i:]]
    residual_Na = [fit_list_Na[n-i] - d_list_Na[n] for n in range(i, len(n_list_Na))]
    lambda_Na = 2000*slope_int[0]
    
    print('Na wavelength = ',2000*slope_int[0],'+/-', 2000*np.sqrt(cov_val[0][0])/np.sqrt(len(d_list_Na[i:])), 'nm')
    plt.figure()
    plt.scatter(n_list_Na[i:], residual_Na, label='Na')
    plt.xlabel('n')
    plt.ylabel('Residual')
    plt.legend(loc = 'upper left')


# Each plots encompass a certain range of data points from a unique starting point to the end of the measurement.
# For example, the exact last half of the data produces good residual with small residual values and no significant pattern. $\lambda_{Na, mean} = 590.28 \pm 0.05 nm$. Last 11 data points produce the expected value as the mean value ($\lambda_{Na, mean} = 589.27 \pm 0.11 nm$) and the corresponding residual plots look good.  Then it is safe to assume that we have to twist more than $\Delta n = 500$ to get a better result (to avoid the effect of hysterisis). If there were more time, I would do the experiment with this condition.

# ## Na Doublet Data: Mar 29, 2023
# 
# Data analysis for $\Delta\lambda_{Na}$

# reading_list_double = [.995, [.28, .3], 0.985, [0.3, 0.32], [0.58, 0.61], 0.72, [0.01, 0.03],[0.29, 0.33], [0.59, 0.62], [0.9, 0.93], 0.125, [0.425, 0.45], [0.71, 0.74], [0.01, 0.04], [0.31, 0.34], 0.86, [0..12, 0.15], [0.425, 0.45], [0.74, 0.77]]

# In[5]:


# after calibration
data_d_list_Na_double = [ [ 1.28-0.995, 1.3-0.995 ], [ 1.3-0.985, 1.32-0.985 ], [ 0.58-0.32, 0.61-0.32], [ 1.01-0.72, 1.03-0.72 ], [ 0.29-0.03, 0.33-0.03 ], [ 0.59-0.33, 0.62-0.33 ], [ 0.9-0.62, 0.93-0.62 ], [ 0.425-0.125, 0.45-0.125 ], [ 0.71-0.45, 0.74-0.45 ], [ 1.01-0.74, 1.04-0.74 ], [ 0.31-0.04, 0.34-0.04], [ 1.12-0.86, 1.15-0.86 ], [ 0.425-0.15, 0.45-0.15 ], [ 0.74-0.45, 0.77-0.45 ] ]


# In[135]:


data_d_list_Na_double


# In[136]:


# Getting uncertainties for each data point
d_list_Na_double = []
u_Na_double = []

for data in data_d_list_Na_double:
    d_list_Na_double.append( (data[0]+data[1])/2 )
    u_Na_double.append( abs(data[0]-data[1])/2 )
    
trial_num = [i+1 for i in range(len(d_list_Na_double))]    


# In[137]:


# getting mean and standard deviation of data set
mean_delta_d = stats.mean(d_list_Na_double)
std_delta_d = stats.stdev(d_list_Na_double)


# In[152]:


plt.figure()
plt.errorbar(trial_num, d_list_Na_double, u_Na_double, label=f'$\Delta$n = 1', fmt='ok')
plt.axhline(mean_delta_d)
plt.axhline(mean_delta_d + std_delta_d, c = 'r', label = f'$\Delta d + 1\sigma$')
plt.axhline(mean_delta_d - std_delta_d, c = 'g', label = f'$\Delta d - 1\sigma$')
plt.xlim(0,15)
plt.xlabel('Trials')
plt.ylabel(r'$\Delta d$')
plt.legend(loc = 'upper right')


# 12 out of 14 data points are within $\pm 1\sigma$. Does not seems to have any pattern.

# In[142]:


# getting normalized residuals
norm_residual_delta_d = [(mean_delta_d - d_list_Na_double[i])/u_Na_double[i] for i in range(len(trial_num)) ]


# In[143]:


plt.figure()
plt.scatter(trial_num, norm_residual_delta_d, label='Na')
plt.axhline(0)
plt.xlabel('Trials')
plt.ylabel('Normalized Residual')
plt.legend(loc = 'upper left')


# In[145]:


chi2 = np.sum((np.array(norm_residual_delta_d))**2)
print(chi2)


# Almost 12 out of $\chi^2 = 21$ came from a single data point (trial 2). Anyway, $\chi^2 = 21$ is comparable to the number of trials. So, the mean value can be accepted. 

# In[168]:


delta_lambda = (589.27**2)/(2*mean_delta_d*1e6)
print(delta_lambda)


# In[172]:


std_err_delta_lambda = np.sqrt((589.27*0.11/(mean_delta_d*1e6))**2 + 0.25*(589.27**2*(std_delta_d*1e6)/(mean_delta_d*1e6)**2)**2)
print(std_err_delta_lambda)


# $\Delta\lambda = 0.59 \pm 0.03 nm$
# The expected value is 0.6nm, which is within the range. 
