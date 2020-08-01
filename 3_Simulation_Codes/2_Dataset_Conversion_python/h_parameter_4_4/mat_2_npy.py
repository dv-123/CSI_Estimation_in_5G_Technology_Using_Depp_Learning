from scipy.io import loadmat

theta_mat = loadmat('theta.mat') # contains all the matab variables generated during program running

from pandas import DataFrame

df = DataFrame(list(theta_mat.items()), columns = ['column1','column2'])

df_1 = df['column2']

theta_val = df_1[[10]].values
r_val = df_1[[6]].values
rxSig_val = df_1[[12]].values

theta = theta_val[0]
r = r_val[0]
rxSig = rxSig_val[0]

import numpy as np

np.save('theta.npy', theta)
np.save('r.npy',r)
np.save('rxSig.npy',rxSig)

# Note: similar program will be used for HMMSE data conversion only the variables
# and the position in the dictionary changes.
