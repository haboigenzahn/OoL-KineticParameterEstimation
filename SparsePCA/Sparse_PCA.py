import numpy as np
import itertools
import pandas as pd
from sklearn.decomposition import SparsePCA
import matlab.engine
# In[1]:

# FILE UPDATE
df = pd.read_csv("../Data/GlyAla_AllData_Recalibrated.csv")
print(df.shape)

# Data matrix; unpacked from dataframe for ease of use
data = df.to_numpy()[:,2:] 
data = np.array(data, dtype=float)


# In[2]: Sparse PCA

# Create new matrix where each column is the reaction species complex associated with that k
# ie, if G + G --k1--> GG, then pcadata column 1 is [G]*[G], and if GG --k2--> G + G, then pcadata column 2 is [GG]
# Species concentrations in the original data matrix are in the same order as in the main file
# species order in data: ['G','A','GG', 'AA', 'GA/AG','GGA/GAG/AGG','GGG','GGGG','AAG/AGA/GAA','AAA']  

pcadata = np.zeros((data.shape[0],22))

pcadata[:,0] = data[:,0]*data[:,0] # k1, G**2
pcadata[:,1] = data[:,2] # k2, GG
pcadata[:,2] = data[:,1]*data[:,1] # k3, A**2
pcadata[:,3] = data[:,3] # k4, AA
pcadata[:,4] = data[:,0]*data[:,1] # k5, A*G
pcadata[:,5] = data[:,4] # k6, AG/GA
pcadata[:,6] = data[:,0]*data[:,4] # k7, AG/GA * G
pcadata[:,7] = data[:,5] # k8, GGA/GAG/AGG
pcadata[:,8] = data[:,0]*data[:,2] # k9, G*GG
pcadata[:,9] = data[:,6] # k10, GGG
pcadata[:,10] = data[:,0]*data[:,6] # k11, G*GGG
pcadata[:,11] = data[:,7] # k12, GGGG
pcadata[:,12] = data[:,2]*data[:,2] # k13, GG*GG
pcadata[:,13] = data[:,7] # k14, GGGG
pcadata[:,14] = data[:,1]*data[:,2] # k15, A*GG
pcadata[:,15] = data[:,5] # k16, GGA/GAG/AGG
pcadata[:,16] = data[:,0]*data[:,3] # k17 G*AA
pcadata[:,17] = data[:,8] # k18, SSG/SGS/GSS
pcadata[:,18] = data[:,1]*data[:,4] #k19, A*AG/GA
pcadata[:,19] = data[:,8] # k20 AAG/AGA/GAA
pcadata[:,20] = data[:,1]*data[:,3] # k21 A*AA
pcadata[:,21] = data[:,9] # k22, AAA

# scale such that the maximum value of each column equals 1 and minimum equals 0
def scale(x, ub, lb, sf = 1):
    m = sf/(ub-lb)
    b = -lb*sf/(ub-lb)

    return m*x+b

# Scale data and find eigenvalues
pcadata_scaled = pcadata.copy()
for i in range(pcadata.shape[1]):
    ub = max(pcadata[:, i])
    lb = min(pcadata[:, i])
    pcadata_scaled[:, i] = scale(pcadata[:, i], ub ,lb)
Σ_scaled = (pcadata_scaled.T@pcadata_scaled)
λ = np.linalg.eigvals(Σ_scaled)

# Using sklearn
transformer = SparsePCA(n_components=22,random_state=0)
Wsparse = transformer.fit(pcadata_scaled).components_

# Using MATLAB
eng=matlab.engine.start_matlab()
Wsparse_matlab=np.array(eng.f(matlab.double(pcadata_scaled.tolist())))
eng.quit()

# In[3]: 
# Print output to console - copy to file as needed 
print("λ: " + np.array2string(λ))
print("\nWsparse from sklearn: " + np.array2string(Wsparse))
print("\nWsparse from MATLAB: " + np.array2string(Wsparse_matlab))
