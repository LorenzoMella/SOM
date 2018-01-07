#############################
#                           #
#   Vector Quantization     #
#                           #
#############################


import numpy as np


euclidean_dist = lambda a,b: np.linalg.norm(a - b, ord=2)


def sq_distances_v(X, W):
    height, width, fan_in = W.shape
    diff = X - W.broadcast_to((height,width,1,fan_in))
    return np.sum(diff*diff, axis=-1)


def voronoi_cells(X, W, dist=euclidean_dist):
    raise NotImplementedError
    max_samples = X.shape[0]
    sq_dists = sq_distances(X, W)
    argmin_dist = np.argmin(sq_dists, axis=-1)
    indexes = np.arange(max_samples).broadcast_to((1, 1, max_samples))
    return np.equals(argmin_dist, indexes)


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