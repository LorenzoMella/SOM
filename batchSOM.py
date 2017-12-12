#===========================================#
#        SOM - Batch Map version            #
#                                           #
#        Author: Lorenzo Mella              #
#        Version: 2017-12-12                #
#    Tested on: Python 3.6.3/numpy 1.13.3   #
#===========================================#

"""
Nothing interesting here YET
"""

import numpy as np
from matplotlib import pyplot
from PCA import covariance_eig
from SOM import batch_dot

height = 50
width = 50

# Algorithm is structured as such:
# Initialize prototypes (the 3-array W)

X = None # some data imported from somewhere
param_shape = (height, width, X.shape[1])
W = np.random.randn(param_shape)

# Find all the distances between prototypes and data-points
# The end-result is another 3-array. The first two indices yield the neuron
# position. The third index spans the distances from said neuron to all
# datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
# (height, width, fan_in)
# np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])

def sq_distances_v(X, W):
    diff = X.T - W[:,:,:,np.newaxis]
    return batch_dot(diff, diff)

# The same behavior should arise from the more elementary code:
def sq_distances_e(X, W):
    height, width = W.shape[:2]
    max_samples = X.shape[0]
    sq_distances = np.empty(shape=(height, width, max_samples, dtype=np.float64)
    for i in range(height):
        for j in range(width):
            # This last loop can be shortened as a matrix product
            for n in range(max_samples):
                sq_distances[i,j,n] = np.dot(X[n,:], W[i,j,:])
    return sq_distances


def voronoi_cells(X, W):
    raise NotImplementedError

# Then we assign neurons to Voronoi Cells. To do this we build a mask of
# argmins. For each neuron it tells (with a 1-in-K encoding) whether its
# prototype is the closest to x or not. The resulting matrix will be sparse
# and it could be saved as such in much the same way as the Matlab Toolbox does.
