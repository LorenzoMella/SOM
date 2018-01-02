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
from time import process_time

from matplotlib import pyplot
from PCA import covariance_eig
from SOM import batch_dot

height = 50
width = 50

# Algorithm is structured as such:
# Initialize prototypes (the 3-array W)

# X = None # some data imported from somewhere
# param_shape = (height, width, X.shape[1])
# W = np.random.randn(param_shape)

# Find all the distances between prototypes and data-points
# The end-result is another 3-array. The first two indices yield the neuron
# position. The third index spans the distances from said neuron to all
# datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
# (height, width, fan_in)
# np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])

def sq_distances_v(X, W):
    height, width, fan_in = W.shape
    diff = X - W[:,:,np.newaxis,:]
    return np.sum(diff*diff, axis=-1)

def sq_distances_t(X, W):
    height, width, fan_in = W.shape
    diff = X - np.tile(W[:,:,np.newaxis,:], (1, 1, X.shape[0], 1))
    return np.sum(diff*diff, axis=-1)

# The same behavior should arise from the more elementary code:
def sq_distances_m(X, W):
    height, width = W.shape[:2]
    max_samples = X.shape[0]
    sq_distances = np.empty(shape=(height, width, max_samples),
                            dtype=np.float64)
    for i in range(height):
        for j in range(width):
            """
            The content of this scope has been tested as equivalent to:
            for n in range(max_samples):
                diff = X[n,:] - W[i,j,:]
                sq_distances[i,j,n] = np.dot(diff, diff)
            """
            diff = X - W[i,j,:]
            sq_distances[i,j,:] = batch_dot(diff, diff)
    return sq_distances


def time_v(max_iter, max_samples):
    W = np.random.randn(50, 49, 11)
    X = np.random.randn(max_samples, 11)
    start = process_time()
    for iteration in range(max_iter):
        sq_distances_v(X,W)
    finish = process_time()
    return finish - start

def time_t(max_iter, max_samples):
    W = np.random.randn(50, 49, 11)
    X = np.random.randn(max_samples, 11)
    start = process_time()
    for iteration in range(max_iter):
        sq_distances_t(X,W)
    finish = process_time()
    return finish - start

def time_m(max_iter, max_samples):
    W = np.random.randn(50, 49, 11)
    X = np.random.randn(max_samples, 11)
    start = process_time()
    for iteration in range(max_iter):
        sq_distances_m(X,W)
    finish = process_time()
    return finish - start

def equal_results(decimal_places=10):
    W = np.random.randn(50, 49, 11)
    X = np.random.randn(10000, 11)
    res_m = sq_distances_m(X,W)
    res_v = sq_distances_v(X,W)
    return np.all(np.absolute(res_m - res_v) < 10**-decimal_places)


def voronoi_cells(X, W):
    """ At this point, return an array of shape (height, width, max_samples)
        whose boolean entry [i, j, n] is True iff X[n,:] belongs to the Voronoi
        Cell of prototype W[i,j,:].
    """
    height, width, fan_in = W.shape
    max_samples = X.shape[0]
    sq_dists = sq_distances_m(X, W)
    argmin_dist = np.argmin(sq_dists, axis=-1)
    indexes = np.arange(max_samples)[np.newaxis, np.newaxis, :]
    mask = np.equals(argmin_dist, indexes)
    return mask

# Then we assign neurons to Voronoi Cells. To do this we build a mask of
# argmins. For each neuron it tells (with a 1-in-K encoding) whether its
# prototype is the closest to x or not. The resulting matrix will be sparse
# and it could be saved as such in much the same way as the Matlab Toolbox does.
