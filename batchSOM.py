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
#from PCA import covariance_eig
from SOM import batch_dot, umatrix
import SOM_data_providers as dp

height = 50
width = 49

# Find all the distances between prototypes and data-points
# The end-result is another 3-array. The first two indices yield the neuron
# position. The third index spans the distances from said neuron to all
# datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
# (height, width, fan_in)
# np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])

def sq_distances_v(X, W):
    height, width, fan_in = W.shape
    diff = X - W[:,:,np.newaxis,:]
    return np.sum(diff**2, axis=-1)

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
    return np.all(np.isclose(res_m, res_v, atol=10**-decimal_places))


def voronoi_cells(X, W):
    """ Return an array of shape (height, width, max_samples)
        whose boolean entry [i,j,n] is True iff X[n,:] belongs to the Voronoi
        Cell of prototype W[i,j,:].
    """
    max_samples = X.shape[0]
    sq_dists = sq_distances_m(X, W)
    argmin_dist = np.argmin(sq_dists, axis=-1)
    indexes = np.arange(max_samples)
    mask = np.equal(argmin_dist[:,:,np.newaxis], indexes)
    return mask


def voronoi_cells_e(X, W):
    max_samples = X.shape[0]
    height, width, _ = W.shape
    sq_dists = sq_distances_m(X, W)
    argmin_dist = np.argmin(sq_dists, axis=-1)
    mask = np.empty(shape=(height, width, max_samples))
    for i in range(height):
        for j in range(width):
            for n in range(max_samples):
                mask[i,j,n] = argmin_dist[i,j] == n
    return mask


def sum_cell(X, mask):
    """ Return an array of shape (height, width, fan_in) whose slice [i,j,:]
        is the vector sum of examples X[n,:] that belong to the Voronoi Cell
        of prototype W[i,j,:].
    """
    return np.dot(mask, X)


def sum_cell_e(X, mask):
    height, width, max_features = W.shape
    max_samples, _ = X.shape
    cell_sum_X = np.zeros(shape=(height, width, max_features))
    for i in range(height):
        for j in range(width):
            for n in range(max_samples):
                cell_sum_X[i,j,:] = ( cell_sum_X[i,j,:]
                                     + (X[n,:] if mask[i,j,n] == 1 else 0) )
    return cell_sum_X


def update_W(X, W):
    mask = voronoi_cells(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    cell_mean_X = sum_cell(X, mask) / cell_num_elems[:,:,np.newaxis]
    # THE FOLLOWING IS THE DENOMINATOR
    # Neighborhoods are unions of adjacent (non diagonal) cells. We are
    # computing them with a toroidal topology, because it's easier and
    # empirically shown as more convenient
    neigh_num_elems = ( cell_num_elems
                        + np.roll(cell_num_elems, shift=-1, axis=0)
                        + np.roll(cell_num_elems, shift=1, axis=0)
                        + np.roll(cell_num_elems, shift=-1, axis=1)
                        + np.roll(cell_num_elems, shift=1, axis=1) )
    
    # neigh_mean_X, INSTEAD, IS THE NUMERATOR
    neigh_weighted_X = cell_num_elems[:,:,np.newaxis] * cell_mean_X
    neigh_mean_X = ( neigh_weighted_X
                     + np.roll(neigh_weighted_X, shift=-1, axis=0)
                     + np.roll(neigh_weighted_X, shift=1, axis=0)
                     + np.roll(neigh_weighted_X, shift=-1, axis=1)
                     + np.roll(neigh_weighted_X, shift=1, axis=1) )
    # Update weights
    return neigh_mean_X / neigh_num_elems[:,:,np.newaxis]


def update_W_e(X, W):
    height, width, max_features = W.shape
    mask = voronoi_cells_e(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    cell_sum_X = sum_cell_e(X, mask)
    cell_mean_X = np.empty((height, width, max_features))
#     for i in range(height):
#         for j in range(width):
#             cell_mean_X[i,j,:] = cell_sum_X[i,j,:] / cell_num_elems[i,j]
    # Neighborhoods are unions of adjacent (non diagonal) cells. We are
    # computing them with a toroidal topology, because it's easier and
    # empirically shown as more convenient
    neigh_num_elems = ( cell_num_elems
                        + np.roll(cell_num_elems, shift=-1, axis=0)
                        + np.roll(cell_num_elems, shift=1, axis=0)
                        + np.roll(cell_num_elems, shift=-1, axis=1)
                        + np.roll(cell_num_elems, shift=1, axis=1) )
    
    # neigh_mean_X, INSTEAD, IS THE NUMERATOR
    neigh_sum_X = ( cell_sum_X
                     + np.roll(cell_sum_X, shift=-1, axis=0)
                     + np.roll(cell_sum_X, shift=1, axis=0)
                     + np.roll(cell_sum_X, shift=-1, axis=1)
                     + np.roll(cell_sum_X, shift=1, axis=1) )
    # Update weights
    W_new = np.empty(W.shape)
    for i in range(height):
        for j in range(width):
            W_new[i,j,:] = neigh_sum_X[i,j,:] / neigh_num_elems[i,j]
    return W_new


if __name__ == '__main__':
    X, labels, _ = dp.polygon_clusters_dataset()
    W = np.random.randn(height, width, X.shape[1])
    
    for t in range(1):
        W1 = update_W(X, W)
        W2 = update_W_e(X, W)
        
    pyplot.figure('U-Matrix and Input Space Scenario')
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')
    pyplot.show()
    