import itertools
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

import torch
from torch.autograd.functional import jacobian 

### Basic usage ###

'''
# instantiate ODE fit
model = ODE(system, df, params)

# fit to data
params = model.fit()

where df has columns [Time, Treatments, S1, ..., SN]
'''

# function to compute ODE gradients
def dZdt(system, Z, t, x, params):
    # compute Jacobian (gradient of model w.r.t. x)
    if str(type(t)) != "<class 'torch.Tensor'>":
        t = torch.tensor(t)
    if str(type(x)) != "<class 'torch.Tensor'>":
        x = torch.tensor(x)
    if str(type(params)) != "<class 'torch.Tensor'>":
        params = torch.tensor(params)  
    if str(type(Z)) == "<class 'torch.Tensor'>":
        Z=Z.detach().numpy()
    J = jacobian(system, (t, x, params))
    Jx = J[1]
    Jp = J[2]
    Jx = Jx.detach().numpy()
    Jp = Jp.detach().numpy()
    # print(Jx)
    # print(Jp)
    # print(Z)
    # compute gradient of model w.r.t. parameters

    return Jx@Z + Jp #@ does matrix multipication so doing Jx times Z

# define function that returns model sensitivity vector
def runODEZ(system, t_eval, x, params):
    # check dimensions
    dim_x = len(x)
    n_params = len(params)
    dim_z = dim_x*n_params

    # if not vectorized, xz will be 1-D
    def dXZ_dt(t, xz):
        # split up x and z
        x = xz[:dim_x]
        Z = xz[dim_x:].reshape(dim_x, n_params)
        dzdt = dZdt(system, Z, t, x, params).reshape(dim_z)
        preds = system(t, x, params)
        if str(type(dzdt)) != "<class 'torch.Tensor'>":
            dzdt = torch.tensor(dzdt)
        if str(type(preds)) != "<class 'torch.Tensor'>":
            preds = torch.tensor(preds)
        dXZdt = torch.cat([preds, dzdt])
        return dXZdt
    # Jacobian of augmented system
    def jac(t, xz):
        t = torch.tensor(t, dtype=float, requires_grad=True)
        xz = torch.tensor(xz, dtype=float, requires_grad=True)
        J = jacobian(dXZ_dt,(t,xz))
        jac = J[1]
        # print(jac)
        return jac.detach().numpy()

    # set initial condition to z equal to zeros
    xz = np.concatenate((x, np.zeros(dim_z)))

    # solve ODE model
    # print(t_eval[-1])
    # print(t_eval)
    # print(xz)
    soln = solve_ivp(dXZ_dt, (0, t_eval[-1]), xz, t_eval=t_eval,
                     jac=jac, method='LSODA')
    t_solver, y = soln.t, soln.y.T
    return y

class ODE:
    def __init__(self, system, df, params, bounds=None, yCOV=None,
                 verbose=True, n_jobs=-1):
        # store experimental data
        # columns: [Treatment name], [Time], [x_1], ..., [x_n]
        # self.system = jit(system)
        self.system = system
        self.df = df
        # make sure params are 1-dimensional
        self.params = np.array(params).ravel()
        # number of parameters
        self.n_params = len(params)
        # bounds on params
        self.bounds = bounds
        # dimension of model output
        self.n_species = df.shape[1]-2

        # runtime parameters
        self.verbose = verbose
        self.n_jobs = n_jobs

        # optionally input expected covariance on output variables
        self.yCOV = yCOV

        # set posterior parameter precision and covariance to None
        self.Beta  = None
        self.A = None
        self.Ainv = None

        # store treatment names
        self.all_treatments = df.Treatments.values
        self.unique_treatments = np.unique(self.all_treatments)
        self.N = df.shape[0]
        self.species_names = df.columns.values[2:]

        # store measured datasets for quick access
        self.data = []
        for treatment in self.unique_treatments:

            # pull community trajectory
            comm_inds = np.in1d(self.df['Treatments'].values, treatment)
            comm_data = self.df.iloc[comm_inds].copy()

            # make sure comm_data is sorted in chronological order
            comm_data.sort_values(by='Time', ascending=True, inplace=True)

            # pull evaluation times
            t_eval = np.array(comm_data['Time'].values, np.float16)

            # pull just the community data
            Y_measured = np.array(comm_data[self.species_names].values, np.float16)

            # append t_eval and Y_measured to data list
            self.data.append([t_eval, Y_measured])

    # Scale data for SSE calculation    
    def scale(self, x, sf = 1):
        Y_scaled=np.zeros(np.shape(x))
        species_index = 0
        
        for col in self.df.columns:
            if col == 'Time' or col == 'Treatments':
                pass
            else:
                ub = max(self.df[col])
                lb = min(self.df[col])
                
                if ub-lb <= 1e-5: # preventing divide by 0 errors
                    Y_scaled[:,species_index] = 0
                else:
                    m = sf/(ub-lb)
                    b = -lb*sf/(ub-lb)
                    Y_scaled[:,species_index] = m*x[:,species_index]+b
                    
                species_index = species_index + 1 # doing this incrementation manually because I don't want it to count Time/Treatments
                
        return Y_scaled


    # MLE of Alpha and Beta parameters
    def update_precision(self):

        # estimate covariance of output variables
        if self.Ainv is None:
            self.yCOV = 0
            self.SSE  = 0
            # loop over each sample in dataset
            for t_eval, Y_measured in self.data:

                # run model using current parameters
                output = runODEZ(self.system, t_eval, Y_measured[0, :], self.params)

                # collect gradients and reshape
                G = np.reshape(output[:, self.n_species:],
                              [output.shape[0], self.n_species, self.n_params])

                # Determine SSE and gradient of SSE
                Y_predicted = output[:, :self.n_species] # dim = T x n_species
                #Y_error = Y_predicted - Y_measured[:output.shape[0],:]
                Y_error = self.scale(Y_predicted) - self.scale(Y_measured[:output.shape[0],:])
                self.yCOV += (Y_predicted-Y_measured).T@(Y_predicted-Y_measured)
                self.SSE  += 1/2*np.sum(Y_error**2)

            # update hyper-parameters
            ### MLE update ###
            alpha = self.n_params / np.dot(self.params, self.params)
            self.Alpha = alpha*np.eye(self.n_params)
            # target precision
            self.Beta  = self.N*np.linalg.inv(self.yCOV)
            self.Beta  = (self.Beta + self.Beta.T)/2
            # compute Bayesian information criterion
            self.BIC = -self.SSE + self.N/2*np.sum(np.log(np.linalg.eigvalsh(self.Beta))) - 1/2*self.n_params*np.log(self.N)
        else:
            ### Evidence based update ###
            gamma = np.sum(np.linalg.eigvals(self.Gamma).real)
            alpha = gamma/np.dot(self.params, self.params)
            beta  = (self.N - gamma) / (2*self.SSE)
            self.lmbda = alpha/beta
            print("Total samples: {}, Effective params: {}, Updated regularization: {:.2e}".format(self.N, int(gamma), self.lmbda))
            self.Alpha = alpha*np.eye(self.n_params)
            self.Beta  = self.N*np.linalg.inv(self.yCOV)
            self.Beta  = (self.Beta + self.Beta.T)/2
            # compute Bayesian information criterion
            self.BIC = -self.NLP + self.N/2*np.sum(np.log(np.linalg.eigvalsh(self.Beta))) - 1/2*self.n_params*np.log(self.N)

        print("BIC {:.3f}".format(self.BIC))
        print("Updated output precision: ")
        print(np.diag(self.Beta))

    def update_covariance(self, update_Hessian=False):
        # update parameter covariance matrix given current parameter estimate
        if update_Hessian:
            self.H = 0
            self.SSE = 0
            self.yCOV = 0
            self.NLP = np.einsum('i,ij,j->',self.params,self.Alpha,self.params)/2

            # loop over each sample in dataset
            for t_eval, Y_measured in self.data:

                # run model using current parameters
                output = runODEZ(self.system, t_eval, Y_measured[0, :], self.params)

                # collect gradients and reshape
                G = np.reshape(output[:, self.n_species:],
                              [output.shape[0], self.n_species, self.n_params])

                # compute Hessian of log-likelihood
                self.H += np.einsum('tki, kl, tlj->ij', G, self.Beta, G)

                # determine SSE and gradient of SSE
                Y_predicted = output[:, :self.n_species] # dim = T x n_species
                #Y_error = Y_predicted - Y_measured[:output.shape[0],:]
                Y_error = self.scale(Y_predicted) - self.scale(Y_measured[:output.shape[0],:])
                if self.Ainv is None:
                    self.yCOV += (Y_predicted-Y_measured).T@(Y_predicted-Y_measured)
                else:
                    self.yCOV += np.einsum('tki,ij,tlj->kl', G, self.Ainv, G) + (Y_predicted-Y_measured).T@(Y_predicted-Y_measured)
                self.NLP += np.einsum('tk,kl,tl->',(Y_predicted-Y_measured),self.Beta,(Y_predicted-Y_measured))/2
                self.SSE += 1/2*np.sum(Y_error*2)
        # Compute Hessian of negative-log-likelihood (NLL)
        self.H = (self.H + self.H.T)/2
        # Compute Hessian and inverse Hessian of negative-log-posterior (NLP)
        self.A = self.Alpha + self.H
        self.Ainv = np.linalg.inv(self.A)
        self.Ainv = (self.Ainv + self.Ainv.T)/2
        self.Gamma = self.H@self.Ainv
    


    def objective(self, params):
        # compute Hessian, covariance of y, sum of squares error
        self.H = 0
        self.yCOV = 0
        self.SSE  = 0
        # compute negative log posterior (NLP)
        self.NLP = np.einsum('i,ij,j->',self.params,self.Alpha,self.params)/2
        # compute gradient of negative log posterior
        self.grad_NLP = self.Alpha@params
        
        # compute gradients of cost function
        for t_eval, Y_measured in self.data:

            # run model using current parameters
            output = runODEZ(self.system, t_eval, Y_measured[0, :], params)
            if output.shape[0] < Y_measured.shape[0]:
                print("Warning: could not compute full trajectory!")

            # collect gradients and reshape
            G = np.reshape(output[:, self.n_species:],
                          [output.shape[0], self.n_species, self.n_params])

            # compute Hessian
            self.H += np.einsum('tki, kl, tlj->ij', G, self.Beta, G)

            # determine SSE and gradient of SSE
            Y_predicted = output[:, :self.n_species] # dim = T x n_species
            Y_error = self.scale(Y_predicted) - self.scale(Y_measured[:output.shape[0],:])
            #Y_error = Y_predicted- Y_measured[:output.shape[0],:]
            
            if self.Ainv is None:
                self.yCOV += (Y_predicted-Y_measured).T@(Y_predicted-Y_measured)
            else:
                self.yCOV += np.einsum('tki,ij,tlj->kl', G, self.Ainv, G) + (Y_predicted-Y_measured).T@(Y_predicted-Y_measured)
            self.NLP += np.einsum('tk,kl,tl->',(Y_predicted-Y_measured),self.Beta,(Y_predicted-Y_measured))/2
            self.SSE += 1/2*np.sum(Y_error*2)

            # sum over time and outputs to get gradient w.r.t params
            self.grad_NLP += np.einsum('tk,kl,tli->i', (Y_predicted-Y_measured), self.Beta, G)
        # Hessian of NLL
        self.H = (self.H + self.H.T)/2
        # Hessian of NLP
        self.A = self.Alpha + self.H
        return self.NLP

    def jacobian(self, params):
        # compute gradient of cost function
        return self.grad_NLP

    def hessian(self, params):
        # compute hessian of NLP
        return self.A

    def callback(self, xk, res=None):
        if self.verbose:
            print("Total weighted fitting error: {:.3f}".format(self.NLP))
        return True

    def fit(self, BIC_tol=1e-3, beta_tol=1e-3):
        # estimate parameters using gradient descent
        convergence = np.inf
        prev_BIC    = 0

        print("\nRunning gradient descent...")
        while convergence > BIC_tol:
            # update Alpha and Beta hyper-parameters
            self.update_precision()
            # fit using updated Alpha and Beta
            if self.bounds is None:
                self.res = minimize(fun=self.objective, x0=self.params,
                           jac=self.jacobian, hess=self.hessian, tol=beta_tol,
                           method='Newton-CG',
                           callback=self.callback)
            else:
                self.res = minimize(fun=self.objective, x0=self.params,
                           jac=self.jacobian, tol=beta_tol,
                           bounds=self.bounds,
                           callback=self.callback)
            if self.verbose:
                print(self.res)
            self.params = self.res.x
            # check convergence
            convergence = np.abs(prev_BIC - self.BIC) / np.max([1,np.abs(prev_BIC)])
            # update BIC
            prev_BIC = np.copy(self.BIC)
            # update covariance
            self.update_covariance()
        print("BIC {:.3f}".format(self.BIC))

    def predict(self, x_test, teval):
        # check if precision has been computed
        if self.Ainv is None:
            self.update_covariance()

        # make predictions given initial conditions and evaluation times
        output = runODEZ(self.system, teval, x_test, self.params)

        # reshape gradient
        G = np.reshape(output[:, self.n_species:],
                       [output.shape[0], self.n_species, self.n_params])

        # compute Hessian over trajectory
        H = np.einsum('tki, kl, tlj->ij', G, self.Beta, G)

        # calculate variance of each output (dimension = [steps, outputs])
        covariance = np.linalg.inv(self.Beta) + np.einsum('tki,ij,tlj->tkl', G, self.Ainv, G)
        variance   = np.diag(np.linalg.inv(self.Beta)) + np.einsum('tki,ij,tkj->tk',  G, self.Ainv, G)
        stdv = np.sqrt(variance)
        return output[:, :self.n_species], stdv, covariance, H

    def search(self, master_df, n_test, n_jobs=None):
        '''
        Master DF represents total design space, D
        n_test is the number of experiments to select from D
        n_jobs is the number of cores on which to process utility function
        '''

        # format data and get experiment names to search over
        all_experiments = master_df.Treatments.values
        unique_experiments = np.unique(all_experiments)

        # initialize list of best experiments
        best_experiments = []
        design_utility   = []
        H = None

        while len(best_experiments) < n_test:
            # score utility for all potential designs
            print("Searching for experiment {}...".format(len(best_experiments)+1))

            # compute utilities in parallel
            if n_jobs is not None:
                experiment_utilities = Parallel(n_jobs=n_jobs)(delayed(self.experiment_utility)(master_df, experiment, H) for experiment in unique_experiments)
            else:
                experiment_utilities = []
                for experiment in unique_experiments:
                    experiment_utilities.append(self.experiment_utility(master_df, experiment, H))

            # unpack experiments and utilities keeping same order
            experiments = np.array([u[0] for u in experiment_utilities])
            utilities   = np.array([np.float(u[1]) for u in experiment_utilities])
            '''plt.hist(utilities, bins=1000)
            plt.show()'''
            # determine order to sort utility from best to worst
            utility_sort = np.argsort(utilities)[::-1]

            for i, (best_experiment, utility) in enumerate(zip(experiments[utility_sort], utilities[utility_sort])):
                # ignore if best utility experiment already in training set
                print("Picked {}\n".format(best_experiment))
                if best_experiment not in best_experiments:
                    best_experiments += [best_experiment]
                    design_utility   += [utility]
                    H = self.expected_hessian(master_df, best_experiments)
                    break
        best_experiments, updated_utilities = self.update_search(master_df, best_experiments, 10)
        return best_experiments, np.append(design_utility, updated_utilities)

    def update_search(self, master_df, best_experiments, n_iterations, n_jobs=-1):
        # update set of best experiments
        best_experiments = np.array(best_experiments)
        design_utility   = self.design_utility(master_df, best_experiments)

        # format data and get experiment names to search over
        all_experiments = master_df.Treatments.values
        unique_experiments = np.unique(all_experiments)

        while True:
            # score utility for all dropouts
            drop_utilities   = [self.design_utility(master_df, best_experiments[best_experiments!=experiment]) for experiment in best_experiments]

            # drop worst experiment
            drop_exp = best_experiments[np.argmax(drop_utilities)]
            best_experiments = best_experiments[best_experiments!=drop_exp]
            print("Dropped {}".format(drop_exp))

            # compute current Hessian
            H = self.expected_hessian(master_df, best_experiments)

            # compute utilities in parallel
            if n_jobs is not None:
                experiment_utilities = Parallel(n_jobs=n_jobs)(delayed(self.experiment_utility)(master_df, experiment, H) for experiment in unique_experiments)
            else:
                experiment_utilities = []
                for experiment in unique_experiments:
                    experiment_utilities.append(self.experiment_utility(master_df, experiment, H))

            # unpack experiments and utilities keeping same order
            experiments = np.array([u[0] for u in experiment_utilities])
            utilities   = np.array([np.float(u[1]) for u in experiment_utilities])
            # determine order to sort utility from best to worst
            utility_sort = np.argsort(utilities)[::-1]

            for i, (best_experiment, utility) in enumerate(zip(experiments[utility_sort], utilities[utility_sort])):
                # ignore if best utility experiment already in training set
                print("Picked {}\n".format(best_experiment))
                if best_experiment == drop_exp:
                    best_experiments = np.append(best_experiments, best_experiment)
                    design_utility   = np.append(design_utility, utility)
                    return np.array(best_experiments), np.array(design_utility)

                if best_experiment not in best_experiments:
                    best_experiments = np.append(best_experiments, best_experiment)
                    design_utility   = np.append(design_utility, utility)
                    break

    def experiment_utility(self, master_df, trial_experiment, H=None):
        # compute utility of a single experiment
        exp_inds = np.in1d(master_df.Treatments.values, np.array(trial_experiment))
        trial_df = master_df.iloc[exp_inds].copy().sort_values(by="Time")
        initial_condition = trial_df.iloc[0, 2:].values
        t_eval = trial_df.Time.values
        pred, var, coV, Hi = self.predict(initial_condition, t_eval)

        # compute design Hessian if previous H is given
        if H is not None:
            Hi += H

        # compute expected information matrix
        Apost = self.A + Hi

        # log of determinant of posterior precision equal to
        # sum of the log of the eigenvalues
        U = np.sum(np.log(np.linalg.eigvalsh(Apost)))

        # if utility is inf or -inf, set equal to zero
        if np.isinf(U) or np.isnan(U):
            print('Warning: numerically unstable utility function')
            U = 0

        return trial_experiment, U.item()

    def expected_hessian(self, master_df, trial_experiments):
        # compute expected Hessian
        H = 0
        for i, trial_experiment in enumerate(trial_experiments):
            # pull experiment info from master df
            exp_inds = np.in1d(master_df.Treatments.values, np.array(trial_experiment))
            trial_df = master_df.iloc[exp_inds].copy().sort_values(by="Time")
            initial_condition = np.array(trial_df.iloc[0, 2:].values, np.float16)
            t_eval = np.array(trial_df.Time.values, np.float16)
            pred, var, coV, Hi = self.predict(initial_condition, t_eval)
            # cumulative hessian
            H += Hi
        return H

    def design_utility(self, master_df, trial_experiments):
        # compute expected Hessian
        H = self.expected_hessian(master_df, trial_experiments)

        # compute expected information matrix
        Apost = self.A + H

        # log of determinant of posterior precision equal to
        # sum of the log of the eigenvalues
        U = np.sum(np.log(np.linalg.eigvalsh(Apost)))

        # if utility is inf or -inf, set equal to zero
        if np.isinf(U) or np.isnan(U):
            print('Warning: numerically unstable utility function')
            U = 0

        return U.item()
