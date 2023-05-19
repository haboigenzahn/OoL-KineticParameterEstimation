READ ME
Kinetic modeling and parameter estimation of a prebiotic peptide reaction network
by Hayley Boigenzahn, Leonardo Gonzalez, Jaron Thompson, Victor Zavala, and John Yin

Parameter estimation of an ODE network based on experimental data. 

May 2023

Corresponding author: john.yin@wisc.edu

--------------------- Environment/Dependencies ----------------------------------------------------
 All dependencies are included in the requirements.txt file. This may include some unnecessary 
	dependencies, since we worked in the Spyder IDE. 

General Points -- 
	- Use Ctrl+F FILE UPDATE to identify points in the code that may enable or 
		disable optional sections, or may have file names that need to be updated.

Imports
	- numpy, itertools, pandas, matplotlib.pyplot, seaborn, scipy, autODE_torch
		scipy.integrate, scipy.optimize, torch, joblib

------------------- Individual file guide ------------------------------------
ParameterFitting-main -- 
	- This is the main file for fitting parameters to experimental data and plotting the results.
	- Reads a .csv file from Data folder 
	- Calls autODE_torch module
	- Performs parameter fitting and plots results. 
	- Optional: The final section of the code executes the DoE method
	- Optional: By starting at section 5.5, it can be used to plot previously fitted systems 
		without redoing teh entire calculation
	
	Output details: 
	- Saves the parameter fit and the results for the covariance matrix in the Params folder
	- Optional: Can save plots of results

autODE_torch --
	- Does all the mathematical heavy lifting for the parameter estimation procedure
	- Generally does not need to be accessed
	
Simulated Data Generator --
	- Given a set of parameters, solve for the concentration vs. time profiles those parameters produce
	- Can parameters from an existing results file in Params, or can be input manually
	- Can input various combinations of initial conditions to simulate the results of
	- Optional: Can add simulated noise to the data
	
	Output details: 
	- Saves the simulated data to the Data folder as a .csv
	
Reduced Networks -- 
	- Alternative definitions for the system function to describe different reaction networks
	
SparsePCA -- 
	- Main file is Sparse_PCA
	- Reads a .csv file from the Data folder
	- Prints eigenvalue list and sparse matrices from MATALB and SciPy SPCA methods to the console
	- Other files are helper mathematical functions
	
Params --
	- Folder containing parameter results and covariance matrices from parameter fitting
	- Parameter files should be three columns x num_parameters - parameter, expected value, and std. dev. 
		Expected value is the fitted parameter mean and std. dev. is the parameter standard deviation.
	- Covariance files a num_parameters x num_parameters matrix, plus the mean parameter estimates in the 
		first non-label column. The numbers within the matrix represent the covariance between different 
		parameter pairings, with their variance shown down the diagonal.
		
Data -- 
	- Folder for storing various data sets for parameter fitting. 
	- Datasets should follow the format 
		'Treatments'	'Time'	'species1'	'species2' ... etc.
	- Datasets should be saved as .csv files with no extra, empty rows of commas at the bottom of the file
	
	
	
