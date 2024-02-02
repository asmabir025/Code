#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(20, 10))
mpl.rc('image', cmap='gray')


# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# In[2]:


mpl.rcParams['text.usetex'] = True


# In[3]:


import numba


# In[4]:


import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import latex
import pickle


# In[5]:


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel



frames = gray(pims.Video('02m02min_DIA.mp4'))


# In[6]:


plt.imshow(frames[0]);


# In[7]:


f = tp.locate(frames[0], 17, invert=True, minmass=1000)
tp.annotate(f, frames[0]);

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');


# In[107]:


f_batch = tp.batch(frames, 11, minmass=200, invert=True, processes=1, engine='numba');


# In[119]:


# tp.quiet()  # Turn off progress reports for best performance
t = tp.link(f_batch, 5, memory=3)


# In[120]:


t1 = tp.filter_stubs(t, 10)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())


# In[121]:


plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass


# In[122]:


plt.figure()
tp.annotate(t1[t1['frame'] == 0], frames[0]);


# In[123]:


plt.figure()
tp.plot_traj(t1);


# In[124]:


d = tp.compute_drift(t1)


# In[125]:


d.plot()
plt.show()


# In[126]:


tm = tp.subtract_drift(t1.copy(), d)


# In[127]:


ax = tp.plot_traj(tm)
plt.show()


# In[128]:


em = tp.emsd(tm, 0.16, 5.01) # microns per pixel = 100/285., frames per second = 24

fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
#ax.set(ylim=(1e-2, 100));


# In[118]:


means = np.array([])
uncs= np.array([])
for i in range(0, 50, 50):
    msd_data = em.values
    times_data = np.array(em.index)
    msd_data1 = msd_data[i:i+20]
    times_data1 = times_data[i:i+20]

    slope, cov = np.polyfit(times_data1, msd_data1, 1, cov=True)
    slope = np.poly1d(slope)

    plt.scatter(times_data1, msd_data1)
    line = [slope[1] * y1 + slope[0] for y1 in times_data1]
    plt.plot(times_data1, line, color="r")
    unc_slope = np.sqrt(cov[1][1])
    unc_T = 2.5

    def calc_av(slope, T):
        A = slope * (1e-6)**2
        R = 8.31
        #a = ((0.474)/2) * 1e-6 # 0.5 um
        #a = (1.354/2) * 1e-6 # 1 um
        a = (1.840/2) * 1e-6 # 2 um

        eta = 9.544e-4
        avogadro = 4*R*T / (6*np.pi*a*eta*A)

        return avogadro

    mean = calc_av(slope[1], 295)
    mean_dS = calc_av(slope[1] + unc_slope, 295)
    mean_dT = calc_av(slope[1], 295 + unc_T)
    dS = mean_dS - mean
    dT = mean_dT - mean

    unc = np.sqrt(dS**2 + dT**2)

    means = np.append(means, mean)
    uncs = np.append(uncs, unc)
    
    print(f"{mean / 1e23:.2f}e23 +/- {unc/ 1e23:.2f}e23")
    
weights = 1.0/uncs**2
WeightedAvg = np.dot(means,weights)/np.sum(weights)
α = np.sqrt(1.0/np.sum(weights))
WeightedAvg, α
print(f"Final result: {WeightedAvg/1e23:.2f}e23 +/- {α/1e23:.2f}e23")


# In[25]:


file = open("0.5m02min_EPI.txt", "wb")
pickle.dump(f_batch, file)

file.close()


# In[ ]:


import pickle

file = open("05m02min_DIA.txt", "rb")
f_batch = pickle.load(file)

file.close()


# In[ ]:




