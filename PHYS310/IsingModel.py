#!/usr/bin/env python
# coding: utf-8

# In[34]:


from datetime import datetime
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'notebook')
from scipy.optimize import curve_fit

mpl.style.use('classic') 
plt.rc('figure', figsize = (8, 6)) # Reduces overall size of figures
plt.rc('axes', labelsize=16, titlesize=14)
plt.rc('figure', autolayout = True) # Adjusts supblot params for new size


# ## For Infinite Temperature Ising Model

# In[ ]:


# user-defined parameters
Lattice_sizes = [10]   # size of square lattice (L^2 atoms in lattice) - aim for 30-50 once your code works
monte_carlo_steps = 8
N_iter = [monte_carlo_steps * Length**2 for Length in Lattice_sizes] # number of iterations to loop through - make this larger once you get your code working
np.random.seed(13)  # as long as you keep the "seed" the same, the "random" numbers will come out the same in repeated runs - useful while testing/debugging

# set up variables for tracking state



L_mlist = {}

for i in range(len(Lattice_sizes)):
    L = Lattice_sizes[i]
    microstate = np.ones((L, L))    # define a 2D LxL array that will hold the microstate in each iteration.  To start with, fill it with all ones (all spin up).
    mlist = np.zeros(N_iter[i])         # initialize a list that will save how the magnetization per spin m changes over time as we iterate
    microstates = []
    microstates.append(microstate.copy())
    
    for n in range(N_iter[i]):
  
        m = np.sum(microstate) / L**2         # calculate the magnetization per spin, m, of the current microstate (this should be 1 at the beginning, will change over time)
        mlist[n] = m     # save magnetization (per spin) of microstate     

        # In each iteration, you choose one location in the lattice, like throwing a dart at a dartboard.
        # Identify which random spin to potentially flip: generate i and j, two random numbers from 0 to L-1
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        # flip that spin - write a line of code to get the current spin at location (i,j) in microstate, and flip it
        microstate[i][j] *= -1
        microstates.append(microstate.copy())
        
    L_mlist[L] = mlist


# In[ ]:


# # Show the microstate at the end of the run
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(microstates[0])
x = np.array(list(range(len(mlist))))
ax2.plot(x / L**2, mlist)
ax2.set_xlabel("Monte Carlo Steps")
ax2.set_ylabel("Net Magnetization")

# add code to label the axes!
ax1.set_xlabel("i")
ax1.set_ylabel("j")


# ## Finite temperature Ising Model 

# In[ ]:


def get_energy(microstate, i, j):
    L = len(microstate[0])

    # setting up boundary conditions
    curr_val = microstate[i][j]
    val_up = microstate[i-1][j]
    val_down = microstate[(i+1)%L][j]
    val_left = microstate[i][j-1]
    val_right = microstate[i][(j+1)%L]
    neighbors = [val_up, val_down, val_left, val_right]
    sum = 0
    for i in range(4):
        sum += neighbors[i] * curr_val
    return sum

def get_Esys(microstate):
    L = len(microstate[0])
    Esys = 0
    for i in range(L):
        for j in range(L):
            Esys += get_energy(microstate, i, j)
    return Esys

def get_delta_E(microstate, i, j):
    return -2 * get_energy(microstate, i, j)


# In[ ]:


def run_sim(T, L, steps):
    N_iter = steps * L**2
    #global microstate # I ran extra 20k MC to get data to determine correlation length.
    # So, I was using the saved last microstate from the previous run to begin.

    microstate = np.ones((L, L))    # define a 2D LxL array that will hold the microstate
    #in each iteration.  To start with, fill it with all ones (all spin up).
    
    mlist = []         # initialize a list that will save how the magnetization per spin 
    #m changes over time as we iterate
    microstate_list = []
    #microstates.append(microstate.copy())

    for n in range(N_iter):


        # Flip a random particle
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        microstate[i][j] *= -1

        # Calculate energy again
        delta_E = get_delta_E(microstate, i, j)



        if delta_E <= 0:
          pass
        else:
          prob = np.exp(-delta_E/T)

          if np.random.random() < prob:
          #  print("particle has flipped")
            pass
          else:
            # The flip does not occur, flip it back
           # print("particle has not flipped")
            microstate[i][j] *= -1
            
        m = np.sum(microstate) / L**2         # calculate the magnetization per spin, m, of the 
        #current microstate (this should be 1 at the beginning, will change over time)
        mlist.append(m)      # save magnetization (per spin) of microstate 

            
#    final_microstates.append(microstate.copy())

        if n % (10*L**2) == 0: # saving a microstate in each 10MC for large system, but 20MC for 
        #small
            microstate_list.append(microstate.copy())

    return mlist, microstate_list


# In[ ]:


# These temperature divisions are based on the needed number of steps verified 
#from test runs
low_temps = [1, 1.25, 1.5, 1.75, 2]
mid_low_temps = [2.1, 2.15, 2.2]
tc_temps = [2.21, 2.22, 2.23, 2.24, 2.25]
tc1_temps = [2.26, 2.27, 2.28, 2.29, 2.3, 2.31, 2.32]
tc2_temps = [2.33, 2.34, 2.35, 2.37, 2.4, 2.43]
mid_high_temps = [ 2.5, 2.55, 2.6, 2.7]
high_temps = [2.8, 2.9, 3, 3.5, 4, 4.5, 5]
temps = low_temps + mid_low_temps + tc_temps + tc1_temps + tc2_temps + mid_high_temps + high_temps


# In[ ]:


mlists = [] # Initiating a list to capture all lists of m values (for all temperatures)
microstates_list = [] # Initiating a list to capture lists of final 100 microstates (for all temperatures)
temps = [1.5]
L = 50 # or L = 20

for i in range(len(temps)):
    
    if temps[i] in low_temps or temps[i] in high_temps:
        steps = 500
    elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
        steps = 5000
    elif temps[i] in tc_temps: 
        steps = 10000 
    elif temps[i] in tc1_temps or temps[i] in tc2_temps:
        steps = 30000

    mlist, microstates = run_sim(temps[i], L, steps)
    
    # Saving mlist and 100 microstates 
    file1 = open(f"L{L}_T{temps[i]}_mlist.txt", "wb")
    pickle.dump(mlist, file1)
    file1.close()
    
    file2 = open(f"L{L}_T{temps[i]}microstates.txt", "wb")
    pickle.dump(microstates, file2)
    file2.close()
    
    mlists.append(mlist)
    microstates_list.append(microstates)
    
    plt.figure()
    plt.xlabel('Monte Carlo Steps')
    plt.ylabel('m')
    label_T = f'T = {temps[i]}'
    
    x = np.array(list(range(len(mlist))))/ L**2  # Divide by L**2 to get monte carlo steps on the x axis
    plt.plot(x, mlist, label= label_T) # Plot the T plot
    img_name = f'L{L}_T{temps[i]}saved.png'
    plt.legend()
    plt.savefig(img_name)
    plt.close()


    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


# In[ ]:


# Creating <m> vs T

L = 50 #or L = 20

averages = [] # list of average m (for last few hundreds/thousands of MCs) 
#for corresponding T
stds = [] # list of standard deviation in m for corresponding T
slopes = [] # list of slopes of last few hundreds/thousands of MCs for m
#vs t plot
uslopes = [] # list of uncertainties in these slopes

for i in range(len(temps)):
    if temps[i] in low_temps or temps[i] in high_temps:
        averaging_step = 100 # Which monte carlo step to average after
    elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
        averaging_step = 1000
    elif temps[i] in tc_temps: 
        averaging_step = 3000
    elif temps[i] in tc1_temps or temps[i] in tc2_temps:
        averaging_step = 10000
    
    cutoff = averaging_step * L**2 # cutoff for averaging

    
    x = np.array(list(range(len(mlists[i])))) / L**2  # Divide by L**2 to get monte carlo steps on the x axis
    
    
    if temps[i] in tc1_temps or temps[i] in tc2_temps:
        averages.append(np.mean(np.abs(mlists[i][cutoff:]))) # Calculate the average of absolute m values near Tc
    else: 
        averages.append(np.mean(mlists[i][cutoff:])) # Calculate the average after the cutoff
    
    stds.append(np.std(mlists[i][cutoff:])) # Calculate the std after the cutoff
    popt, pcov = curve_fit(f,x[cutoff:],mlists[i][cutoff:])
    slopes.append(popt[0])
    uslopes.append(np.sqrt(pcov[0][0]))

for i in range(len(temps)):
    if temps[i] in low_temps or temps[i] in high_temps:
        steps = 500
    elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
        steps = 5000
    elif temps[i] in tc_temps: 
        steps = 10000 
    elif temps[i] in tc1_temps or temps[i] in tc2_temps::
        steps = 30000
        
    if temps[i] in low_temps or temps[i] in high_temps:
        averaging_step = 100 # Which monte carlo step to average after
    elif temps[i] in mid_low_temps:
        averaging_step = 1000
    elif temps[i] in tc_temps: 
        averaging_step = 3000
    elif temps[i] in tc1_temps or temps[i] in mid_high_temps or temps[i] in tc2_temps:
        averaging_step = 10000

    print(f"T={temps[i]}: {averages[i]:.6f} +/- {stds[i]/np.sqrt((steps - averaging_step)*L**2):.6f}\tSlope: {slopes[i]:.5f} +/- {uslopes[i]:.5f}") # Dividing by sqrt(N), not sure if correct

plt.figure()
plt.scatter(temps, averages, label = 'L=50')
plt.xlabel('Temperature (T)')
plt.ylabel('<m>')
plt.legend()


# In[ ]:


# Runnning extra 2000 MC steps to get data for correlation length calculation

mlists = []

L = 20 # or L = 50
steps = 2000

for i in range(len(temps)):

    
    file2 = open(f"L{L}_T{temps[i]}microstates.txt", "rb")
    microstates_cp = pickle.load(file2)
    
    file2.close()

    microstate = microstates_cp[-1]


    mlist2, microstates2 = run_sim(temps[i], L, steps)
    
    file1 = open(f"L{L}_T{temps[i]}_mlistextra.txt", "wb")
    pickle.dump(mlist2, file1)
    file1.close()
    
    file2 = open(f"L{L}_T{temps[i]}microstatesfinal.txt", "wb")
    pickle.dump(microstates2, file2)
    file2.close()
    
    mlists.append(mlist2)
    
    plt.figure()
    plt.xlabel('Monte Carlo Steps')
    plt.ylabel('m')
    label_T = f'T = {temps[i]}'
    
    x = np.array(list(range(len(mlist2))))/ L**2  # Divide by L**2 to get monte carlo steps on the x axis
    plt.plot(x, mlist2, label= label_T) # Plot the T plot
    img_name = f'L{L}_T{temps[i]}savedextra.png'
    plt.legend()
    plt.savefig(img_name)
    plt.close()


    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


# ## g(m) calculation 

# In[ ]:


def sigma_i_im(microstate, m):
    '''Determines average < sigma_i * sigma_{i+m}  > for all atoms in the lattice 
    for a definite m at a single (equilibrated) time step.
    Parameters:
    - m; int; 0 < m < L/2 +1 
    - microstate; 2D array
    '''
    #gettting column and row length
    rows = len(microstate) 
    cols = len(microstate[0])

    tot_sigma_im = 0 #initiating sum of sigma_i * sigma_{i+m}  
    tot_m = 0 # initiating sum of sigma_m 
    tot_i = 0
    for i in range(rows):
        for j in range(cols):
     
            curr_val = microstate[i][j]
            tot_i += curr_val
            sum_im = 0 # initiating sum of individual sigma_i * sigma_{i+m}  
            sum_m = 0
            
            '''To get the sum from the bottom rows of the central atom'''
            prev = i - m # top most row 
            for k in range(i-m, i, 1): # topmost row to one bottom the atom
               
                diff= k - prev # to calculate difference in column number for the central atom and the boundary atoms
                col_list = [(j-diff), (j+diff)] # only two columns per row because of diamond shape
                
                val = microstate[(k%rows)][col_list[0]%cols]
                sum_m += val
                sum_im += curr_val*val # Adding < sigma_i * sigma_{i+m}  >
                
                if col_list[0] != col_list[1]: # to account for the other column except for the bottom row
                    val = microstate[k%rows][col_list[1]%cols]
                    sum_m += val
                    sum_im += curr_val*val 

                 
            '''To get the sum from the top rows of the central atom including its row'''   
            prev = i+m # central atom's row
            for k in range(i, i+m+1, 1): # central attom row to top row

                diff= prev - k # to calculate difference in column number for the central atom and the boundary atoms
                col_list = [(j-diff), (j+diff)]

                val = microstate[k%rows][col_list[0]%cols]
                sum_m += val
                sum_im += curr_val*val 

                if col_list[0] != col_list[1]:
                    val = microstate[k%rows][col_list[1]%cols]
                    sum_m += val
                    sum_im += curr_val*val 

            tot_sigma_im += sum_im/(m*4) # To get average of all neighbors; m*4 because of diamond shape
            tot_m += sum_m/(m*4)
   
    return tot_sigma_im/(rows*cols), tot_i/(rows*cols), tot_m/(rows*cols)


# In[ ]:


def g(m, microstate_list):
    '''Calculates average < sigma_i * sigma_{i+m}  > using sigma_i_im function for
    last 100 (equilibrated) Monte Carlo steps and determines the average across all MC 
    steps.
    Parameters:
    - m; int; 0 < m < L/2 +1 
    - microstate_list; list of 2D arrays
    '''
    g_list = []
    for i in range(len(microstate_list)):
        sigma_val, i_val, im_val =  sigma_i_im( microstate_list[i], m ) 
        
        g_list.append(  (sigma_val - im_val*i_val)/(1 - i_val*im_val)  )
        
    return np.mean( g_list ), statistics.stdev( g_list )


# In[ ]:


L = 50 # or L = 20
m_list = [i for i in range(1, L//2+1)]

for T in temps:    
   
    file = open(f"L{L}_T{T}microstates.txt", "rb") #opening the file with the correct L and T
    #which contains last 100/200 microstates 
    microstates = pickle.load(file)
    file.close()
    
    # determing g(m) for  0 < m < L/2+1 

    g_avg_list = [] # list of g(m) values
    g_std_list = [] # list of standard deviations in g(m)
    for i in range(1, L//2+1 ):
        mean_g, std_g = g(i, microstates)
        g_avg_list += [ mean_g ]
        g_std_list += [ std_g ]
        
    g_err_list = [g_std_list[i]/np.sqrt(100) for i in range(len(g_std_list))]
    g_info = [ g_avg_list, g_err_list ]
    
    file = open( f"L{L}_T{T}_g.txt", 'wb' ) # saving g(m) and uncertainty in g(m)
    pickle.dump(g_info, file)
    file.close()


# ## Correlation length calculation

# In[ ]:


L = 50 #or L = 20
m_list = [ i for i in range(1, L//2+1)] # list for the number of interactions
m_list_curve = np.linspace(1, L//2+1, 250) # continuous numbers in x-asis for curve fitting 


# In[ ]:


def f(m, xi, u, k):
    '''fitting function for m and g(m)
    '''
    return k/(np.power(m, u)) * np.exp(-m/xi)


# In[ ]:


low_temps = [2]
mid_low_temps = [2.1, 2.15, 2.2]
tc_temps = [2.21, 2.22, 2.23, 2.24, 2.25]
tc1_temps = [2.26, 2.27, 2.28, 2.29, 2.3, 2.31, 2.32]
tc2_temps = [2.33, 2.34, 2.35, 2.37, 2.4, 2.43]
mid_high_temps = [2.47, 2.5, 2.55, 2.6, 2.7]
high_temps = [2.8, 2.9, 3, 3.5, 4, 4.5, 5]
temps = low_temps + mid_low_temps + tc_temps + tc1_temps + tc2_temps + mid_high_temps + high_temps

xi_list = []
for T in temps:
    file = open( f'L{L}_T{T}_g.txt', 'rb' ) # opening the saved file of g and g_err list
    g_info = pickle.load(file)
    file.close()

    g_avg_list = g_info[0]
    g_err_list = np.array(g_info[1])

    # Define the initial guess for the parameters xi and u
    initial_guess = [1.0, 1.0, 1.0]
    last = 12 # or last = 5 if L = 20
    # Last defines how many m values we are curve-fitting


    # Fit the function to the data using curve_fit
    popt, pcov = curve_fit(f, m_list[:last], g_avg_list[:last], sigma=g_err_list[:last], p0=initial_guess)
    xi_list.append(popt[0]) # correlation length

    # Print the optimal values for the parameters xi and u
    print("Optimal values for xi, u, and k:", popt)

    # Plot the data and the fitted curve
    plt.figure()
    plt.errorbar(m_list, g_avg_list, yerr=g_err_list, label=f'L={L}, T={T}', fmt="x")
    plt.plot(m_list_curve, f(np.array(m_list_curve), *popt), label='fit')
    plt.ylabel("g(m)")
    plt.xlabel("m")
    plt.legend()
    plt.savefig(f'L{L}_T{T}_gm_fit.png')
    plt.show()
    plt.close()


# In[ ]:


Tc = 2.275 # or Tc = 2.4 for L = 20
plt.figure()
plt.scatter(temps, xi_list, label=f'$L={L}$')
plt.axvline(Tc, c='r')
plt.xlabel(r'$T$')
plt.ylabel(r'$\xi$')
plt.legend()


# In[ ]:


# Plotting residuals for specific temperature g(m) vs m
residuals = (f(np.array(m_list), *popt) - np.array( g_avg_list  ))
plt.figure()
plt.errorbar( m_list, residuals, g_err_list, fmt = 'ok' )


# In[ ]:


#chi^2
np.sum((residuals/np.array( g_err_list ))**2)


# In[ ]:


# normalized residual plotting
plt.figure()
plt.errorbar( m_list, (residuals/np.array( g_err_list )), fmt = 'ok', label = f'L={L}, T={T}' )
plt.xlabel('m')
plt.ylabel('Normalized Residuals')
plt.axhline(2)
plt.axhline(-2)
plt.ylim(-5, 5)
plt.legend(loc=2)
plt.savefig(f'L{L}_T{T}_normresiduals.png')


# In[ ]:


# Saving correlation length list
file = open(f'xi_vs_T_L{L}.txt', 'wb')
pickle.dump([temps, xi_list], file)
file.close()


# In[35]:


# These temperature divisions are based on the needed number of steps verified 
#from test runs
low_temps = [1, 1.25, 1.75, 2]
mid_low_temps = [2.1, 2.15, 2.2]
tc_temps = [2.21, 2.22, 2.23, 2.24, 2.25]
tc1_temps = [2.27, 2.28, 2.29, 2.31, 2.32]
tc2_temps = [2.33, 2.34, 2.35, 2.37, 2.4, 2.43]
mid_high_temps = [ 2.5, 2.55, 2.6, 2.7]
high_temps = [2.8, 2.9, 3, 3.5, 4, 4.5, 5]
temps = low_temps + mid_low_temps + tc_temps + tc1_temps + tc2_temps + mid_high_temps + high_temps


# In[36]:


# Creating <m> vs T

L = 50 #or L = 20

averages = [] # list of average m (for last few hundreds/thousands of MCs) 
#for corresponding T
stds = [] # list of standard deviation in m for corresponding T

for i in range(len(temps)):
    if temps[i] in low_temps or temps[i] in high_temps:
        averaging_step = 100 # Which monte carlo step to average after
        step = 500
    elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
        averaging_step = 500
        step = 5000
    elif temps[i] in tc_temps or temps[i] in tc2_temps: 
        averaging_step = 1000
        step = 10000
    elif temps[i] in tc1_temps:
        averaging_step = 10000
        step = 30000
    
    cutoff = averaging_step * L**2 # cutoff for averaging
    
    filename = f'L{L}_T{temps[i]}_mlist.txt'
    file = open(filename, 'rb')
    mlist  =  pickle.load(file)
    file.close()
    
    #x = np.array(list(range(len(mlist)))) / L**2  # Divide by L**2 to get monte carlo steps on the x axis
    
    
    if temps[i] in tc1_temps or temps[i] in tc2_temps or temps[i] in tc_temps:
        averages.append(np.mean(np.abs(mlist[cutoff: step*L**2]))) # Calculate the average of absolute m values near Tc
        stds.append(np.std(np.abs(mlist[cutoff: step*L**2]))/np.sqrt((step - averaging_step)*L**2)) # Calculate the std after the cutoff
        print(np.mean(np.abs(mlist[cutoff: step*L**2])))
    else: 
        averages.append(np.mean(mlist[cutoff: step*L**2])) # Calculate the average after the cutoff
        stds.append(np.std(mlist[cutoff: step*L**2])/np.sqrt((step - averaging_step)*L**2)) # Calculate the std after the cutoff
    print(averages[i], stds[i])
#     popt, pcov = curve_fit(f,x[cutoff:],mlists[i][cutoff:])
#     slopes.append(popt[0])
#     uslopes.append(np.sqrt(pcov[0][0]))

# for i in range(len(temps)):
#     if temps[i] in low_temps or temps[i] in high_temps:
#         steps = 500
#     elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
#         steps = 5000
#     elif temps[i] in tc_temps: 
#         steps = 10000 
#     elif temps[i] in tc1_temps or temps[i] in tc2_temps::
#         steps = 30000
        
#     if temps[i] in low_temps or temps[i] in high_temps:
#         averaging_step = 100 # Which monte carlo step to average after
#     elif temps[i] in mid_low_temps:
#         averaging_step = 1000
#     elif temps[i] in tc_temps: 
#         averaging_step = 3000
#     elif temps[i] in tc1_temps or temps[i] in mid_high_temps or temps[i] in tc2_temps:
#         averaging_step = 10000

#     print(f"T={temps[i]}: {averages[i]:.6f} +/- {stds[i]/np.sqrt((steps - averaging_step)*L**2):.6f}\tSlope: {slopes[i]:.5f} +/- {uslopes[i]:.5f}") # Dividing by sqrt(N), not sure if correct


# In[ ]:


slope_list = []
x_intercept = []
unc_slope_list = []
y_intercept = []
for i in range(2, len(averages)-2):
    coeffs,cov = (np.polyfit(temps[i-2:i+3], averages[i-2:i+3], 1, cov=True))
    m,b = coeffs
    x_intercept.append( -b/m )
    y_intercept.append( b )
    slope_list.append(m)
    unc_slope_list.append(np.sqrt(cov[0][0]))


# In[ ]:


index_max_slope = slope_list.index(min(slope_list))
print(temps[index_max_slope], slope_list[index_max_slope], unc_slope_list[index_max_slope])


# In[ ]:


for i in range(len(slope_list)):

    print(x_intercept[i], slope_list[i], unc_slope_list[i], temps[i], y_intercept[i])


# In[ ]:


slope_list = []
for i in range(len(averages)-1):
    slope_list.append((averages[i+1]-averages[i])/(temps[i+1]/temps[i]))


# In[ ]:


index_max_slope = slope_list.index(min(slope_list))
print(temps[index_max_slope], slope_list[index_max_slope])


# In[ ]:


temps[12]


# In[37]:


Tc = r'$2.27 \pm 0.02$'

plt.figure()
plt.errorbar(temps, averages, xerr= stds, fmt='.k', animated=0)
# plt.plot( temps, slope_list[index_max_slope]*np.array(temps)+ y_intercept[index_max_slope], c='r'  )
# plt.axhline(0)
# plt.xlabel('$T$', fontsize=20, fontweight='bold')
# plt.ylabel(r'$\mathbf{\langle m \rangle}$', fontsize=20, fontweight='bold')
# plt.xlim(0.75, 5.25)
# plt.ylim(-0.1, 1.05)
# label = r'$L={}$'.format(L)
# plt.text(4, 0.9, label, fontsize=20, bbox=dict(facecolor='white', edgecolor='black'))
# plt.text(2.3, 0.631699, f'$T_c = ${Tc}', fontsize=15, fontweight='bold')
# plt.savefig(f'L{L}_mvsTupdated.png')


# In[ ]:


slope_list[index_max_slope]*np.array(temps)+ y_intercept[index_max_slope]


# In[ ]:


m_Tc = [slope_list[index_max_slope]*temps[i] + y_intercept[index_max_slope] for i in range(len(temps))]


# In[ ]:


slope_list[index_max_slope]*2.27 + y_intercept[index_max_slope]


# In[ ]:


# These temperature divisions are based on the needed number of steps verified 
#from test runs
low_temps = [1, 1.25, 1.75, 2]
mid_low_temps = [2.1, 2.15, 2.2]
tc_temps = [2.21, 2.22, 2.23, 2.24, 2.25, 2.26]
tc1_temps = [2.27, 2.28, 2.29, 2.31, 2.32]
tc2_temps = [2.33, 2.34, 2.35, 2.37, 2.4]
mid_high_temps = [2.43, 2.47, 2.5, 2.55, 2.6, 2.7]
high_temps = [2.8, 2.9, 3, 3.5, 4, 4.5, 5]
temps = low_temps + mid_low_temps + tc_temps + tc1_temps + tc2_temps + mid_high_temps + high_temps


# In[ ]:


# Creating <m> vs T

L = 20 #or L = 20

averages = [] # list of average m (for last few hundreds/thousands of MCs) 
#for corresponding T
stds = [] # list of standard deviation in m for corresponding T

for i in range(len(temps)):
    if temps[i] in low_temps or temps[i] in high_temps:
        averaging_step = 100 # Which monte carlo step to average after
        step = 500
    elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
        averaging_step = 500
        step = 5000
    else: 
        averaging_step = 1000
        step = 10000
#     elif temps[i] in tc1_temps:
#         averaging_step = 10000
#         step = 30000
    
    cutoff = averaging_step * L**2 # cutoff for averaging
    
    filename = f'L{L}_T{temps[i]}_mlist.txt'
    file = open(filename, 'rb')
    mlist  =  pickle.load(file)
    file.close()
    
    #x = np.array(list(range(len(mlist)))) / L**2  # Divide by L**2 to get monte carlo steps on the x axis
    
    
    if temps[i] in tc1_temps or temps[i] in tc2_temps or temps[i] in tc_temps:
        averages.append(np.mean(np.abs(mlist[cutoff: step*L**2]))) # Calculate the average of absolute m values near Tc
        stds.append(np.std(np.abs(mlist[cutoff: step*L**2]))/np.sqrt((step - averaging_step)*L**2)) # Calculate the std after the cutoff

    else: 
        averages.append(np.mean(mlist[cutoff: step*L**2])) # Calculate the average after the cutoff
        stds.append(np.std(mlist[cutoff: step*L**2])/np.sqrt((step - averaging_step)*L**2)) # Calculate the std after the cutoff
    print(averages[i], stds[i])
#     popt, pcov = curve_fit(f,x[cutoff:],mlists[i][cutoff:])
#     slopes.append(popt[0])
#     uslopes.append(np.sqrt(pcov[0][0]))

# for i in range(len(temps)):
#     if temps[i] in low_temps or temps[i] in high_temps:
#         steps = 500
#     elif temps[i] in mid_low_temps or temps[i] in mid_high_temps:
#         steps = 5000
#     elif temps[i] in tc_temps: 
#         steps = 10000 
#     elif temps[i] in tc1_temps or temps[i] in tc2_temps::
#         steps = 30000
        
#     if temps[i] in low_temps or temps[i] in high_temps:
#         averaging_step = 100 # Which monte carlo step to average after
#     elif temps[i] in mid_low_temps:
#         averaging_step = 1000
#     elif temps[i] in tc_temps: 
#         averaging_step = 3000
#     elif temps[i] in tc1_temps or temps[i] in mid_high_temps or temps[i] in tc2_temps:
#         averaging_step = 10000

#     print(f"T={temps[i]}: {averages[i]:.6f} +/- {stds[i]/np.sqrt((steps - averaging_step)*L**2):.6f}\tSlope: {slopes[i]:.5f} +/- {uslopes[i]:.5f}") # Dividing by sqrt(N), not sure if correct


# In[ ]:


slope_list = []
x_intercept = []
unc_slope_list = []
y_intercept = []
for i in range(2, len(averages)-2):
    coeffs,cov = (np.polyfit(temps[i-2:i+3], averages[i-2:i+3], 1, cov=True))
    m,b = coeffs
    x_intercept.append( -b/m )
    y_intercept.append( b )
    slope_list.append(m)
    unc_slope_list.append(np.sqrt(cov[0][0]))
    
index_max_slope = slope_list.index(min(slope_list))
print(temps[index_max_slope], slope_list[index_max_slope], unc_slope_list[index_max_slope])

for i in range(len(slope_list)):

    print(x_intercept[i], slope_list[i], unc_slope_list[i], temps[i], y_intercept[i])


# In[ ]:


Tc = r'$2.340 \pm 0.010$'

plt.figure()
plt.errorbar(temps, averages, xerr= stds, fmt='.k', animated=0)
plt.plot( temps, slope_list[index_max_slope]*np.array(temps)+ y_intercept[index_max_slope], c='r'  )
plt.axhline(0)
plt.xlabel('$T$', fontsize=20, fontweight='bold')
plt.ylabel(r'$\mathbf{\langle m \rangle}$', fontsize=20, fontweight='bold')
plt.xlim(0.75, 5.25)
plt.ylim(-0.1, 1.05)
label = r'$L={}$'.format(L)
plt.text(4, 0.9, label, fontsize=20, bbox=dict(facecolor='white', edgecolor='black'))
plt.text(2.35, 0.57525, f'$T_c = ${Tc}', fontsize=15, fontweight='bold')
plt.savefig(f'L{L}_mvsTupdated.png')


# In[ ]:


for i in range(len(averages)):
    print(averages[i], temps[i])


# In[20]:


L = 50
L1 = 20
T = 3.5
T1 = 3.5
label_L = f'L={L}'
label_L1 = f'L={L1}'

filename = f'L{L}_T{T}_mlist.txt'
file = open(filename, 'rb')
mlist  =  pickle.load(file)
file.close()

filename = f'L{L1}_T{T1}_mlist.txt'
file = open(filename, 'rb')
mlist1  =  pickle.load(file)
file.close()


# In[27]:


n1 = 500
n = n1
x1 = np.array(list(range(len(mlist1[:n1*L1**2]))))/ L1**2  # Divide by L**2 to get monte carlo steps on the x axis
fig1, axs1 = plt.subplots(1, 2)
axs1[0].plot(x1, mlist1[:n1*L1**2])
# axs1[0].text(6000, 0.0, label_L1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
# axs1[0].text(6000, -0.25, label_T1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
xstr = 't(MC Steps)'
axs1[0].set_xlabel(xstr, fontsize=20)
axs1[0].set_ylabel(r'$m$', fontsize=20)
# axs1[0].axhline(0.8, xmin=0.1,ls='--', c='r')
x = np.array(list(range(len(mlist[:n*L**2]))))/ L**2
axs1[1].plot(x, mlist[:n*L**2]) # Plot the T plot
#axs1[1].set_xticks([])
# axs1[1].axhline(0.8, c='r')
plt.gcf().set_size_inches(10,5)

img_name = f'L{L1}_T{T1}final.png'
print(img_name)
plt.savefig(img_name)


# In[4]:


n = 500
x = np.array(list(range(len(mlist[:n*L**2]))))/ L**2  # Divide by L**2 to get monte carlo steps on the x axis

plt.figure()
plt.plot(x, mlist[:n*L**2])
xstr = r'MC Steps'
plt.rcParams["font.family"] = "cursive"
plt.xlabel(xstr, fontsize=20, fontweight='bold')
plt.ylabel(r'm', fontsize=20, fontweight='bold')
# plt.axhline(0.55, xmin=0.2, ls='--', c='r')

plt.axhline(0.55, c='r', linewidth=1)
plt.ylim(0.4, 1.0)
# plt.gcf().set_size_inches(10,4.5)

img_name = f'L{L}_T{T}final.png'

plt.savefig(img_name)


# In[11]:


n = 30000
x = np.array(list(range(len(mlist[:n*L**2]))))/ L**2  # Divide by L**2 to get monte carlo steps on the x axis

plt.figure()
plt.plot(x, mlist[:n*L**2])
xstr = r'MC Steps'
plt.xlabel(xstr, fontsize=20, fontweight='bold')
plt.ylabel(r'm', fontsize=20, fontweight='bold')
# plt.axhline(0.55, xmin=0.2, ls='--', c='r')

# plt.axhline(0.55, c='r', ms=4)
plt.ylim(-1.0, 1.0)


img_name = f'L{L}_T{T}finalfull.png'

plt.savefig(img_name)


# In[ ]:


label_T = f'$T = {T}$'
label_L = f'$L = {L}$'
# n = len(mlist)
n = 50*L**2
x = np.array(list(range(len(mlist[:500*L**2]))))/ L**2  # Divide by L**2 to get monte carlo steps on the x axis
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, mlist[:500*L**2])
axs[0].text(100, 0.99, label_L, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
axs[0].text(300, 0.99, label_T, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
xstr = r'$MC-Steps$'
axs[0].set_xlabel(xstr, fontsize=20)
axs[0].set_ylabel(r'$m$', fontsize=20)
axs[0].axhline(0.96, xmin=0.2, ls='--', c='r')
x = np.array(list(range(len(mlist[:n]))))/ L**2
axs[1].plot(x, mlist[:n]) # Plot the T plot
axs[1].axhline(0.96, c='r')
axs[1].set_ylim(0.94, 1.0)
plt.gcf().set_size_inches(10,4.5)

img_name = f'L{L}_T{T}final.png'

plt.savefig(img_name)


# In[ ]:


L1 = 20
T1 = 2.25
filename1 = f'L{L1}_T{T1}_mlist.txt'
file1 = open(filename1, 'rb')
mlist1  =  pickle.load(file1)
file1.close()


# In[ ]:


label_T1 = f'$T = {T1}$'
n1 = len(mlist1)
n1 = 400*L1**2
x1 = np.array(list(range(len(mlist1[:n1]))))/ L1**2  # Divide by L**2 to get monte carlo steps on the x axis
plt.figure()
plt.plot(x1, mlist1[:n1], label= label_T1) # Plot the T plot
# img_name = f'L{L}_T{temps[i]}saved.png'
plt.legend()
# plt.savefig(img_name)


# In[ ]:


label_T1 = f'$T = {T1}$'
label_L1 = f'$L = {L1}$'
# n = len(mlist)
n1 = 100*L1**2
x1 = np.array(list(range(len(mlist1[:10000*L1**2]))))/ L1**2  # Divide by L**2 to get monte carlo steps on the x axis
fig1, axs1 = plt.subplots(1, 2)
axs1[0].plot(x1, mlist1[:10000*L1**2])
axs1[0].text(6000, 0.0, label_L1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
axs1[0].text(6000, -0.25, label_T1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
xstr1 = r'$MC-Steps$'
axs1[0].set_xlabel(xstr, fontsize=20)
axs1[0].set_ylabel(r'$m$', fontsize=20)
axs1[0].axhline(0.8, xmin=0.1,ls='--', c='r')
x1 = np.array(list(range(len(mlist1[:n1]))))/ L1**2
axs1[1].plot(x1, mlist1[:n1]) # Plot the T plot
axs1[1].axhline(0.8, c='r')
plt.gcf().set_size_inches(10,5)

img_name = f'L{L1}_T{T1}final.png'

plt.savefig(img_name)


# In[8]:


# # Show the microstate at the end of the run
L1 = 50
L = 20
T1 = 2.25
T = 2.33
label_L = f'L={L}'
label_L1 = f'L={L1}'
label_T = f'T={T}'
label_T1 = f'T={T1}'

filename = f'L{L}_T{T}microstatesfinal.txt'
file = open(filename, 'rb')
microstates  =  pickle.load(file)
file.close()

filename1 = f'L{L1}_T{T1}microstatesfinal.txt'
file1 = open(filename1, 'rb')
microstates1  =  pickle.load(file1)
file1.close()

fig, axs = plt.subplots(1,2)
axs[0].imshow(microstates[-1])
axs[0].set_xlabel(r'Column')
axs[0].set_ylabel(r'Row')
# axs[0].text(0, -1.25, label_L, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
# axs[0].text(-1.5, L+4, label_T, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
axs[1].imshow(microstates1[-1])
# axs[1].text(0, -2.5, label_L1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
# axs[1].text(0, L1+10, label_T1, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
plt.gcf().set_size_inches(8,5)
# add code to label the axes!
# plt.xlabel(r'Column')
# plt.ylabel(r'Row')
# plt.text(L//2, L+2, label_L, fontsize=18, bbox=dict(facecolor='silver', edgecolor='black'))
# plt.text(L+2, 5, label_T, fontsize=18, bbox=dict(facecolor='white', edgecolor='black'))
plt.savefig(f'T={T}T1={T1}_microstate.png')


# In[41]:


filename1 = f'L_50_mvsT.txt'
file1 = open(filename1, 'rb')
info = pickle.load(file1)
file1.close()


# In[33]:


for i in range(len(mlist1[1])):
     if mlist1[0][i]==2.34:
            print(mlist1[1][i])


# In[45]:


info[2]


# In[ ]:




