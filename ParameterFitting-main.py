#!/usr/bin/env python
# coding: utf-8

# In[0]:

import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from scipy.stats import linregress, spearmanr

from autODE_torch  import *


# In[1]:

# Define ODE (time, x, params)
# must return a JAX numpy array
def system(t,y,r):
    #REACTIONS - ODE reaction model

    # Parameter system for network in Fig. 1

    # k1 = 2G -> GG, k2 = GG -> 2G
    # k3 = 2A -> AA, k4 = AA -> 2A
    # k5 = A+G -> AG/GA, k6 = AG/GA -> G + A
    # k7 = AG/GA + G -> GAG/AGG/GGA, k8 = GAG/AGG/GGA -> AG/GA + G
    # k9 = G + GG -> GGG, k10 = GGG -> GG + G
    # k11 = GGG+G -> GGGG, k12 = GGGG -> GGG + G
    # k13 = GG+GG -> GGGG, k14 = GGGG -> GG + GG
    # k15 = GG + A -> GGA/AGG, k16 = GGA/AGG -> GG + A
    # k17 = AA + G -> AAG/GAA, k18 = AAG/GAA -> AA + G
    # k19 = GA/AG + A -> GAA/AGA/AAG, k20 = GAA/AGA/AAG -> GA/AG + A
    # k21 = AA + A -> AAA, k22 = AAA -> AA + A
        
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
      
    if str(type(dGdt)) == "<class 'torch.Tensor'>":
        res = torch.cat([dGdt.flatten(), dAdt.flatten(), dGGdt.flatten(),
                          dAAdt.flatten(), dAGdt.flatten(), dAGGdt.flatten(),
                          dGGGdt.flatten(), dGGGGdt.flatten(), dAAGdt.flatten(),
                          dAAAdt.flatten()])
    else:
        res = np.array([dGdt, dAdt, dGGdt, dAAdt, dAGdt, dAGGdt, dGGGdt, 
                        dGGGGdt, dAAGdt, dAAAdt]) 

    return res

# Scale data for parity plot
# This function does not affect model fitting, the autode_torch_scaled contains a duplicate function that does scale the data during fitting
# Scaled such that the maximum value of each species column equals 1 and minimum equals 0
def scale(x, ub, lb, sf = 1):
    # x is the array to scale
    # ub is the upper bound to scale by
    # lb is the lower bound to scale by
    # sf is an optional scaling factor
    if ub-lb <= 1e-5:
        m = 0
        b = 0
    else:
        m = sf/(ub-lb)
        b = -lb*sf/(ub-lb)

    return m*x+b

# In[2]:
    
# Load data and set up model conditions
    
# Read input data file
# filename will be used in names output files
# FILE UPDATE
filename = "SimulatedTest"
df = pd.read_csv("Data/{}.csv".format(filename))
print(df.shape)

# Define initial parameter guess
params = 5.0*np.ones(22)

# Define bounds on params
bounds = [(0.0, 10.0) for _ in range(len(params))]


# In[3]:

# Fit model to data

# Instantiate ODE fit 
model = ODE(system, df, params=params, bounds=bounds)

# Fit to data 
# BIC_tol and beta_tol are parameters of the minimization function
# BIC_tol is the convergence criterion - the program looks for a model with BIC < BIC_tol
# beta_tol terminates the program when multiple iterations are very close together, meaning it is no longer improving
# Smaller BIC_tol and beta_tol values may make the program take longer to run
model.fit(BIC_tol=1e-3, beta_tol=1e-5)


# In[4]:

# Save Parameter results to DataFrame
param_names = np.array(["k"+str(i+1) for i in range(model.n_params)])
params = model.params
params_std = np.sqrt(np.diag(model.Ainv))
df = pd.DataFrame()
df['Parameter'] = param_names
df['Expected value'] = params 
df['Std. Dev.'] = params_std

df.to_csv("Params/{}_Params.csv".format(filename), index=False)

df.head(len(params))

# In[5]:

# Save Covariance results to DataFrame
df_COV = pd.DataFrame()
df_COV["Param name"] = param_names
df_COV["Param value"] = model.params
for j, param_name in enumerate(param_names):
    df_COV[param_name]  = model.Ainv[:, j]
    
df_COV.to_csv("Params/{}_COV.csv".format(filename), index=False)

# In[5.5]: For recreating plots from previous runs without rerunning the entire fitting procedure
# # Run cell [0] and cell [1] with the imports and system first

# # FILE UPDATE
# # Import previous parameter file
# param_filename = 'GlyAla_10expts'
# df_params = pd.read_csv('Params/{}_Params.csv'.format(param_filename))
# params = df_params['Expected value'].tolist()

# # Load data used to generate the above parameters originally
# df_test = pd.read_csv("Data/{}.csv".format(param_filename))

# # Recalculate model - this is necessary when running only the plotting section of the script
# # It can take a minute, but is significantly faster than refitting from scratch
# bounds = [(0.0, 10.0) for _ in range(len(params))]
# model = ODE(system, df_test, params=params, bounds=bounds)
# model.update_precision()
# model.update_covariance(True)

# In[6]:
    
# Calculate sensitivity eigenvalues for sloppiness analysis
sensitvity_eigs = np.linalg.eigvals(model.H)

# In[7]:

# Plotting - compare model and experimental values

model.update_precision()
model.update_covariance(True)

# FILE UPDATE
df_test = pd.read_csv("Data/{}.csv".format(filename))

all_treatments = df_test.iloc[:, 0].values
species_names = df_test.columns.values[2:]
unique_treatments = np.unique(all_treatments)

# Save all predictions and true values
preds = []
stdvs = []
true  = []

preds_byspecies = [ [] for _ in range(len(species_names)) ]
stdvs_byspecies = [ [] for _ in range(len(species_names)) ]
true_byspecies = [ [] for _ in range(len(species_names)) ]


# Prepare the trajectories used to visualize parameter standard deviation on the model prediction plots
# pull a random community trajectory
for treatment in unique_treatments:
    comm_inds = np.in1d(df_test['Treatments'].values, treatment)
    comm_data = df_test.iloc[comm_inds].copy()

    # make sure comm_data is sorted in chronological order
    comm_data.sort_values(by='Time', ascending=True, inplace=True)
    tspan = comm_data.Time.values

    # pull just the community data
    output_true = comm_data[species_names].values

    # run model using parameters
    x_test = np.copy(output_true[0, :])

    # increase teval
    t_eval = np.linspace(0, tspan[-1])
    steps = len(t_eval)

    # test full community
    numspecies = df_test.shape[1]-2
    output, _, covariance, _ = model.predict(x_test, tspan)
    output_long, _, covariance_long, _ = model.predict(x_test, t_eval)

    # plot the results
    for i in range(numspecies):
        plt.figure(figsize=(9, 9))
        ylim = 0
        out = output[:,i]
        out_long = output_long[:, i]
        std = np.sqrt(covariance[:, i, i])
        std_long = np.sqrt(covariance_long[:, i, i])
        out_true = output_true[:, i]
        
        preds.append(out)
        stdvs.append(std)
        true.append(out_true)
        
        preds_byspecies[i].append(out)
        stdvs_byspecies[i].append(std)
        true_byspecies[i].append(out_true)
        
        # Plot true data as points and model predictions as a line, and fill in uncertainty interval
        ylim = np.max(np.array([np.max(out), np.max(out_true)]) + np.max(std))
        plt.scatter(tspan, out_true, label="True" + str(i+1), color='C{}'.format(i), marker='o', s=75)
        plt.plot(t_eval, out_long, label="Predicted species " + str(i+1), color='C{}'.format(i))
        plt.fill_between(t_eval, out_long-std_long, out_long+std_long, color='C{}'.format(i), alpha=0.2)

        plt.xlabel("Time (days)", fontsize=16)
        plt.ylabel("Concentration", fontsize=16)
        plt.ylim([0, np.min([ylim, 1])])
        plt.title("Treatment {}, Prediction of {}".format(treatment, species_names[i]))
        
        # Save figure output to file system
        # Warning! - This can accumulate many figures very quickly
        #plt.savefig("Figures/Fit_{}_{}.png".format(species_names[i].split("/")[0], treatment), dpi=100)
        
        plt.show()
        plt.close()


# In[8]:

    
# Create parity plot (true data vs model predictions)

def flatten(l):
    return [item for sublist in l for item in sublist]

# Species names for plot labeling
species = ['G','A','GG', 'AA', 'GA/AG','GGA/GAG/AGG','GGG','GGGG','AAG/AGA/GAA','AAA'];   

y_true = np.concatenate(true).ravel()
y_std = np.concatenate(stdvs).ravel()
y_pred = np.concatenate(preds).ravel()

plt.figure(figsize=(9,9))
OneToOne = np.linspace(np.min(y_true), np.max(y_true), 100)
plt.plot(OneToOne, OneToOne, 'k', alpha=.5, label="Perfect fit: R=1")

lr_stats = linregress(y_true, y_pred)
rho, rho_p = spearmanr(y_true, y_pred) 
y_fit = OneToOne*lr_stats.slope + lr_stats.intercept

plt.plot(OneToOne, y_fit, 'k--', alpha=.65, label=r"Model fit: R={:.3f}".format(lr_stats.rvalue))
for i, y in enumerate(preds_byspecies):
    plt.errorbar(flatten(true_byspecies[i]), flatten(y), yerr=flatten(stdvs_byspecies[i]), linewidth=3, capsize=4, linestyle='none', marker='o',alpha=0.7, label=species[i])

plt.legend(fontsize=16, bbox_to_anchor=(1,1))

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("True", fontsize=16)
plt.ylabel("Predicted (y = {:.2f}x + {:.2f})".format(lr_stats.slope, lr_stats.intercept), fontsize=16)

# Save figure output to file system 
#plt.savefig("Figures/{}Fit.png".format(filename), dpi=100)

plt.show()
plt.close()

# In[9]:
    
# Scaled parity plot   

preds_scaled = [ [] for _ in range(len(species_names)) ]
stdvs_scaled = [ [] for _ in range(len(species_names)) ]
true_scaled = [ [] for _ in range(len(species_names)) ]

# scale data in the same manner as autODE_torch
for i, s in enumerate(true_byspecies):
    ub = max(flatten(s)) # upper and lower bound for scaling come from the measured (experimental) data
    lb = min(flatten(s))
    for j, x in enumerate(s):
        true_scaled[i].append(scale(x,ub,lb))
        preds_scaled[i].append(scale(preds_byspecies[i][j],ub,lb))
        stdvs_scaled[i].append(scale(stdvs_byspecies[i][j],ub,lb))

plt.figure("scaled parity", figsize=(9,9))
OneToOne = np.linspace(np.min(flatten(flatten(true_scaled))), np.max(flatten(flatten(true_scaled))), 100)
plt.plot(OneToOne, OneToOne, 'k', alpha=.5, label="Scaled Parity Plot\nPerfect fit: R=1")

lr_stats = linregress(flatten(flatten(true_scaled)), flatten(flatten(preds_scaled)))
rho, rho_p = spearmanr(flatten(flatten(true_scaled)), flatten(flatten(preds_scaled))) 
y_fit = OneToOne*lr_stats.slope + lr_stats.intercept

plt.plot(OneToOne, y_fit, 'k--', alpha=.65, label=r"Model fit: R={:.3f}".format(lr_stats.rvalue))
for i, y in enumerate(preds_scaled):
    plt.errorbar(flatten(true_scaled[i]), flatten(y), yerr=flatten(stdvs_scaled[i]), linewidth=3, capsize=4, linestyle='none', marker='o',alpha=0.7, label=species[i])

plt.legend(fontsize=16, bbox_to_anchor=(1,1))

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("True", fontsize=16)
plt.ylabel("Predicted (y = {:.2f}x + {:.2f})".format(lr_stats.slope, lr_stats.intercept), fontsize=16)

# Save figure output to file system 
#plt.savefig("Figures/{}ScaledFit.png".format(filename), dpi=100)

plt.show()
plt.close()

# In[10]:
    
# Functions for plotting covariance ellipses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

def confidence_ellipse(p1, p2, cov, ax, n_std = 1):    
    #ax.set_aspect("equal")
    
    # correlation and p1, p2 std. dev.
    p = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    scale_p1 = np.sqrt(cov[0, 0]) * n_std
    scale_p2 = np.sqrt(cov[1, 1]) * n_std

    # create ellipse object
    ellipse = Ellipse((p1, p2),
        width=scale_p1 * np.sqrt(1 + p) * 2,
        height=scale_p2 * np.sqrt(1 - p) * 2, 
        angle=45,
        edgecolor='k', 
        facecolor='silver')

    # add plot
    ax.add_artist(ellipse)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return ax

def histogram(p, var, ax, n_std=1):
    support = np.linspace(0, 10)
    ax.plot(support, norm.pdf(support, p, n_std*var**.5))
    ax.axvline(x=p, c='k', linestyle='--', alpha=.5)
    ax.set_xlim(0, 10)
    return ax


# In[11]:
    
# Plot covariance ellipses

param_names = np.array(["k"+str(i+1) for i in range(model.n_params)])
n_params = len(params)

fig, axs = plt.subplots(n_params, n_params, figsize=(32,24))
k = 0
for i in range(n_params):
    for j in range(n_params):
        p1 = model.params[i]
        p2 = model.params[j]
        cov = model.Ainv[[[i, j], [j, j]], [[i, i], [i, j]]] 

        k += 1
        if i != j:
            ax = confidence_ellipse(p1, p2, cov, axs[j, i])
        else:
            ax = histogram(p1, cov[0,0], axs[i, i])
        if j == n_params-1:
            ax.set_xlabel(param_names[i], fontsize=12)
        if i == 0:
            ax.set_ylabel(param_names[j], fontsize=12)

plt.tight_layout()

# Save figure output to file system 
#plt.savefig("Figures/{}Covariance.png".format(filename), dpi=100)

plt.show()
plt.close()

# In[12]:
# Calculate the MSE, with and without scaling
    
y_true = np.concatenate(true).ravel()
y_pred = np.concatenate(preds).ravel()

scaled_ypred = np.array(flatten(flatten(preds_scaled)))
scaled_ytrue = np.array(flatten(flatten(true_scaled)))

mse = ((y_true - y_pred)**2).mean(axis=None)  

scaledmse = ((scaled_ytrue-scaled_ypred)**2).mean(axis=None)

# In[13]: Get experimental design suggestions

# # Note that this code does not automatically remove experiments that are already included in the data
# # So if the possible experiments include conditions that have already been tested, they may be selected again

# # Load file containing possible experiment suggestions
# # The more possible experiments are included, the longer this will take to run
# FILE UPDATE
master_df = pd.read_csv('Data/Initial_conditions_Gly_Ala.csv')


n_tests = 20 # number of tests to choose from the available options
n_jobs = -1 # number of cores used on the computer to parallelize the job; -1 uses all available cores
suggested_experiments = model.search(master_df, n_tests, n_jobs)