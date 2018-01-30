#===========================================#
#        SOM - Batch Map version            #
#        Testing scripts                    #
#                                           #
#        Author: Lorenzo Mella              #
#===========================================#


import numpy as np
from batchSOM import *


def equal_results(decimal_places=10):
    """ Current version: first three tests are passed. Updates still yield
        different results.
    """
    X, _, _ = dp.polygon_clusters_dataset()
    W = np.random.randn(50, 49, X.shape[1])
    # Check whether the two squared-distance algorithms are sufficiently close
    sq_dist_v = sq_distances_v(X,W)
    sq_dist_m = sq_distances_m(X,W)
    sq_dist_equal = np.all( np.isclose(sq_dist_m, sq_dist_v,
                                       atol=10**-decimal_places) )
    # Check that the two methods compute the Voronoi Cells identically
    vor_v = voronoi_cells(X, W)
    vor_e = voronoi_cells_e(X, W)
    vor_equal = np.all( np.equal(vor_e, vor_v) )
    # Check sum_cells equivalence
    sum_X_v = sum_cell(X, vor_v)
    sum_X_e = sum_cell_e(X, vor_e)
    sum_cell_equal = np.all( np.isclose(sum_X_v, sum_X_e,
                                       atol=10**-decimal_places) )
    # Check one iteration of the algorithm with both methods
    W1 = W
    W2 = np.copy(W)
    W1 = update_W(X, W1)
    W2 = update_W_e(X, W2)
    update_equal = np.isclose(W1, W2, atol=10**-5)
    return sq_dist_equal, vor_equal, sum_cell_equal, update_equal


def sum_cell_e(X, mask):
    """ Same as sum_cell but more elementary. For testing purposes.
        Current version of sum_cell does the same as this to 10 decimal places.
    """
    height, width, _ = mask.shape
    max_samples, max_features = X.shape
    cell_sum_X = np.zeros(shape=(height, width, max_features))
    for i in range(height):
        for j in range(width):
            for n in range(max_samples):
                if mask[i,j,n] == True:
                    cell_sum_X[i,j,:] = cell_sum_X[i,j,:] + X[n,:]
    return cell_sum_X


def voronoi_cells_e(X, W):
    """ Elementary version of voronoi_cells. For testing purposes.
    """
    max_samples, _ = X.shape
    height, width, _ = W.shape
    # Build a matrix whose [p,n] entry is the sq_distance between p-th
    # prototype (linearized order) and the n-th datapoint
    prototype_sample_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    # For each X[n,:] choose the closest prototype and register its linearized
    # index
    closest_prototype = np.empty(max_samples)
    for n in range(max_samples):
        closest_prototype[n] = np.argmin(prototype_sample_dists[:,n])
    num_prototypes, _ = prototype_sample_dists.shape
    # Now build a mask expanding each linearized index into a 1-in-K
    # representation
    mask = np.empty(shape=(num_prototypes, max_samples), dtype=np.bool)
    for n in range(max_samples):
        for p in range(num_prototypes):
            mask[p,n] = ( closest_prototype[n] == p )
    return mask.reshape(height, width, max_samples)


def update_W_smooth_e(X, W, sigma2=16.0):
    """ Elementary version of update_W_smooth, for testing purposes.
    """
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    weighted_sum_X = np.zeros((height, width, max_features))
    weight_sum = np.zeros((height, width))
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    i_win, j_win = np.unravel_index(winning_neurons, dims=(height, width))
    for i in range(height):
        for j in range(width):
            for n in range(max_samples):
                output_sq_dist = (i - i_win[n])**2 + (j - j_win[n])**2
                h = np.exp( -0.5 * output_sq_dist / sigma2 )
                weighted_sum_X[i,j,:] += h * X[n,:]
                weight_sum[i,j] += h
    return weighted_sum_X / weight_sum[:,:,np.newaxis]
