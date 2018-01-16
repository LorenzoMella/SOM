#############################
#                           #
#   Vector Quantization     #
#                           #
#############################


import numpy as np


euclidean_dist = lambda a,b: np.linalg.norm(a - b, ord=2)


def sq_distances_v(X, W):
    """ Compute an array of squared distances between any prototype in W
        and any sample vector in X:
          output[i1, ..., i_K-1, n] = sq_dist( X[n,:], W[i1, ..., i_K-1, :] )
    """
    diff = X - W[:,:,np.newaxis,:]
    return np.sum(diff**2., axis=-1)


def voronoi_cells(X, W, dist=euclidean_dist):
    max_samples = X.shape[0]
    sq_dists = sq_distances_v(X, W)
    argmin_dist = np.argmin(sq_dists, axis=-1)
    indexes = np.arange(max_samples)
    mask = np.equal(argmin_dist[:,:,np.newaxis], indexes)
    return mask


def best_centroids(X, cells, dist=euclidean_dist):
    raise NotImplementedError


def avg_reconstruction_error(X, W, dist=euclidean_dist):
    raise NotImplementedError


def VQ(X, W):
    while True:
        cells = voronoi_cells(X, W)
        W_new = best_centroids(X, cells)
        if np.any(W != W_new):
            W = W_new
        else:
            break
    return W
