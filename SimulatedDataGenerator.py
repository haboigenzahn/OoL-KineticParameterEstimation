"""

Solve an ODE system, providing parameters and initial conditions
Used to generate simulated data to fit parameters to

"""

# In[1]: 
    
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# In[2]: 
    
# Declare the ODE model to be integrated forward
def system(t,y,r):
    # REACTIONS - ODE reaction model
        
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22 = r
    G, A, GG, AA, AG, AGG, GGG, GGGG, AAG, AAA = y
    
    dGdt = (-2*k1*G**2 + 2*k2*GG - k5*A*G + k6*AG - k7*AG*G + k8*AGG - k9*G*GG + k10*GGG - k11*G*GGG + k12*GGGG - k17*G*AA + k18*AAG)
    dAdt = (-2*k3*A**2 + 2*k4*AA - k5*A*G + k6*AG - k15*A*GG + k16*AGG - k19*A*AG + k20*AAG - k21*A*AA + k22*AAA)
    dGGdt = (k1*G**2 - k2*GG - k9*G*GG + k10*GGG - 2*k13*GG**2 + 2*k14*GGGG - k15*A*GG + k16*AGG)
    dAAdt = (k3*A**2 - k4*AA - k17*G*AA + k18*AAG - k21*A*AA + k22*AAA)
    dAGdt = (k5*A*G - k6*AG - k7*AG*G + k8*AGG - k19*A*AG + k20*AAG)
    dAGGdt = (k7*G*AG - k8*AGG + k15*A*GG - k16*AGG)
    dGGGdt = (k9*G*GG - k10*GGG - k11*G*GGG + k12*GGGG)
    dGGGGdt = (k11*G*GGG - k12*GGGG + k13*GG**2 - k14*GGGG)
    dAAGdt = (k17*G*AA - k18*AAG + k19*A*AG - k20*AAG)
    dAAAdt = (k21*A*AA - k22*AAA)
    
    dydt = ([dGdt, dAdt, dGGdt, dAAdt, dAGdt, dAGGdt, dGGGdt, dGGGGdt, dAAGdt, dAAAdt])
    
    return dydt
 
# In[3]: Set up parameters and time points   
   
# Either import an existing parameter file, or manually select the parameters

# # FILE UPDATE
# # Import an existing parameter file
# filename = "SimulatedTest"
# df_params = pd.read_csv("Params/{}_Params.csv".format(filename))
# params = df_params.loc[:,'Expected value'].tolist()

# Manual parameter selection
params = [6, 4, 5, 4, 6, 4, 5, 2, 4, 1, 5, 1, 5, 1, 4, 2, 4, 1, 5, 1, 2, 1] # SimulatedTest true parameters

# Pick the number of points to measure from the time course, and how long the time course should continue
# Ensure that numpoints and totaltime are equal for experimentally measured parameters to make the time units align
numpoints = 5
totaltime = 5
tspan = [0,totaltime]
t_eval = [(totaltime/numpoints)*a for a in list(range(numpoints))]

# Specify species labels
species = ['G','A','GG', 'AA', 'GA/AG','GGA/GAG/AGG','GGG','GGGG','AAG/AGA/GAA','AAA'];


# In[4]: Set initial conditions

y0s = ([0,    0.1,    0,    0, 0, 0, 0, 0, 0, 0], 
        [0.01,    0.09,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.025,    0.075,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.05,    0.05,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.075,    0.025,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.09,    0.01,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.1,    0,    0,    0, 0, 0, 0, 0, 0, 0], # original 7 initial conditions ends here
        [0.1,    0.1,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.07,    0.1,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.1,    0.07,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.1,    0.09,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.09,    0.1,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.09,    0.09,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.05,    0,    0,    0, 0, 0, 0, 0, 0, 0],
        [0,    0.05,    0,    0, 0, 0, 0, 0, 0, 0],
        [0.025,    0.025,    0,    0, 0, 0, 0, 0, 0, 0])

# # Extra y0s for increased artificial data
# # Start by adding 0.05 and up for equal concentrations of each monomer
# y0s_extra = np.tile(np.linspace(0.05, 0.1, 6).transpose(), (2,1)).transpose()
# y0s_extra = np.concatenate((y0s_extra, np.zeros((6,8))), axis = 1)
# y0s = np.concatenate((y0s, y0s_extra), axis=0)

# # Start adding n randomly selected initial points that are between 0 and 0.1 M for each amino acid
# n = 3 # number of extra random samples to add onto the previous 16 selected ones
# y0s_extra = np.concatenate((np.random.randint(100,size=(n,2))/1000, np.zeros((n,8))), axis = 1)
# y0s = np.concatenate((y0s, y0s_extra), axis=0)

# In[5]: Solve the system
# Simulate outcomes for multiple y0 conditions

# allsols is a list of length y0s, each entry is two lists - the sol.t list and sol.y list for that y0 solution
allsols_t = []
allsols_y = []
for i, y0 in enumerate(y0s):
    # integrate the ODE
    sol = solve_ivp(system, t_span=tspan, y0=y0, args=[params], t_eval=t_eval)
    # save the results of the ODE evaluation
    allsols_t.append(sol.t)
    allsols_y.append(sol.y)

# flatten results for easier processing
flatsols_y = np.concatenate(allsols_y, axis=1).T

# In[6]: 
# Create a dataframe in the correct format for ParameterFitting-main and save the data file

newdata = {}
for i, s in enumerate(species):
    newdata[s] = flatsols_y[:,i]
    
# Create index label list for treatments
treatments = []
for i in range(len(y0s)):
    treatment_index = np.ones(len(t_eval))*(i)
    treatments.append(treatment_index)
treatments = [item for sublist in treatments for item in sublist]
    
df_results = pd.DataFrame(data=newdata)
df_results.insert(0, "Time", np.concatenate(allsols_t))
df_results.insert(0, "Treaments", treatments)


# In[6.5]: 
# Save the data as a csv file in the format readable by ParameterFitting-main.py

# # FILE UPDATE
filename = "GeneratedTest"
df_results.to_csv("Data/{}.csv".format(filename), index=False)
    
# In[7]: 

# Plot the results

# plot by initial condition -- all species on one plot
for idx, y0 in enumerate(y0s):
    plt.figure(idx)
    plt.title("initial condition " + str(idx))
    for j in range(len(y0)):
        plt.plot(allsols_t[idx],allsols_y[idx][j].T, label=species[j]) 
        plt.legend(bbox_to_anchor=(1,1))

plt.show()
plt.close()

# plot by species -- show all y0s on one plot
for sdx, s in enumerate(species):
    plt.figure(s)
    plt.title(s)
    for idx, y0 in enumerate(y0s):
        plt.plot(allsols_t[idx],allsols_y[idx][sdx].T) 
        plt.scatter(allsols_t[idx],allsols_y[idx][sdx].T) 
        
plt.show()
plt.close()

# In[8]: Optional - add noise to the simulated results
    
sd = 0.15 # standard deviation of the noise
noise = np.random.normal(0, sd, flatsols_y.shape)
noisydata = flatsols_y + flatsols_y*noise

# set anything that the noise made negative to 0, since there are no negative concentrations in experimental data
noisydata[noisydata<0] = 0

# # Optional - replace flatsols_y and allsols_y with the noisy version to be able to run the plotting/file save cells with it
#flatsols_y = noisydata
# allsols_y = []
# for i in range(int(len(treatments)/len(t_eval))):
#     allsols_y.append(noisydata[i*len(t_eval):i*len(t_eval)+len(t_eval),:].T)    
     
# In[8.5]: 
# Save the data as a csv file in the format readable by ParameterFitting-main.py

# # FILE UPDATE
# noisyname = "GeneratedTest_n"+str(sd*100)
# df_results.to_csv("Data/{}.csv".format(noisyname), index=False)