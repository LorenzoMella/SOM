#################################################
#                                               #
#    Utility functions and global parameters    #
#                                               #
#    Author: Lorenzo Mella                      #
#                                               #
#################################################


# Library Imports
from numba import jit
import numpy as np
# Project imports
from SOM_test_common import test_timer


@jit(nopython=True)
def batch_dot(a, b):
    """Array of dot products over the fastest axis of two arrays. Precisely,
       assuming that a and b have K+1 matching axes,
       batch_dot(a, b)[i1, ..., iK] = a[i1, ..., iK, :].dot(b[i1, ..., iK, :])

    """
    # assert a.shape == b.shape # not understood by numba
    return np.sum(a*b, axis=-1)

"""
Scores are used to compare the neuronal responses to an input x. Not the best
choice of name, but the neuron with the lowest score wins. Examples of scores
are squared distances (between weights w and x) or inner products w*x. If the
weights are subsequently normalized, there should be no difference between said
kinds of score.
"""
compute_scores_v = batch_dot

def compute_sq_distances_v(W, x):
    diff = W - x
    return batch_dot(diff, diff)


# Elementary versions of scoring functions for testing. They are more
# transparent; it's easier to see they do the right thing, despite being slow.
# To be used for comparison with the BLAS-based versions.


def compute_scores_elem(W, x):
    max_rows, max_cols, fan_in = W.shape
    scores = np.zeros(shape=[max_rows, max_cols])
    for i in range(max_rows):
        for j in range(max_cols):
            scores[i,j] = np.dot(W[i,j,:], x)
    return scores


@jit(nopython=True)
def compute_sq_distances_m(W, x):
    """Computes the distances between all prototypes and a single datapoint x
    using a non-vectorized algorithm. For checking correctness of sq_distances_v.
    
    Args:
    W (ndarray: (height, width, max_features)): the prototypes
    x (ndarray: (max_features,)): the datapoint

    Returns:
    (ndarray: (height, width)): a matrix of distances
    """
    max_rows, max_cols, fan_in = W.shape
    dists = np.zeros(shape=(max_rows, max_cols))
    for i in range(max_rows):
        for j in range(max_cols):
            diff = W[i,j,:] - x
            dists[i,j] = np.dot(diff, diff)
    return dists


@jit(nopython=True, fastmath=True)  # fastmath works
def compute_sq_distances_elem(W, x):
    """Computes the distances between all prototypes and a single datapoint x
    using a non-vectorized algorithm. For checking correctness of sq_distances_v.
    
    Args:
    W (ndarray: (height, width, max_features)): the prototypes
    x (ndarray: (max_features,)): the datapoint

    Returns:
    (ndarray: (height, width)): a matrix of distances
    """
    max_rows, max_cols, max_features = W.shape
    dists = np.zeros(shape=(max_rows, max_cols))
    for i in range(max_rows):
        for j in range(max_cols):
            for f in range(max_features):
                diff = W[i, j, f] - x[f]
                dists[i, j] += diff * diff
    return dists


def sq_distances_v(X, W):
    """ Computes the distances between all prototypes and a single datapoint x.
    Vectorized version.
    
    Parameters:
    W (ndarray): the prototypes. shape=(height, width, max_features).
    x (ndarray): the datapoint. shape=(max_features,).

    Returns:
    (ndarray): a matrix of distances. shape=(height, width).

    """
    diff = X - W[..., np.newaxis, :]
    return np.sum(diff**2, axis=-1)


@jit(nopython=True, fastmath=True)
def sq_distances_elem(X, W):
    """ Computes the distances between all prototypes and a single datapoint x
    Partially vectorized version.
    
    Notes:
        Fastest thus far. The speed here is likely to be caused by the
        computation arrangement: ndarray objects contain row-major C arrays,
        and the computations are all performed iterating on the fastest axes first.
        No copies should therefore be involved in this process.
    
    Args:
        W (ndarray): the  prototypes.
              shape: (height, width, max_features)
        X (ndarray): the datapoints.
              shape: (max_samples, max_features)

    Returns:
        (ndarray): an array of distances per lattice position.
        shape: (height, width, max_samples)
    """
    #assert X.dtype == W.dtype
    height, width, _ = W.shape
    max_samples, max_features = X.shape
    sq_distances = np.zeros(shape=(max_samples, height, width), dtype=X.dtype)
    for n in range(max_samples):
        for i in range(height):
            for j in range(width):
                for f in range(max_features):
                    diff = X[n, f] - W[i, j, f]
                    sq_distances[n, i, j] += diff * diff
    return np.transpose(sq_distances, (1, 2, 0))


def sq_distances_m(X, W):
    """ Computes the distances between all prototypes and a single datapoint x
    Partially vectorized version.
    
    Notes:
        Fastest thus far. The speed here is likely to be caused by the
        computation arrangement: ndarray objects contain row-major C arrays,
        and the computations are all performed iterating on the fastest axes first.
        No copies should therefore be involved in this process.
    
    Args:
        W (ndarray): the  prototypes.
              shape: (height, width, max_features)
        x (ndarray): the datapoint.
              shape: (max_features,)

    Returns:
        (ndarray): a matrix of distances.
        shape: (height, width)
    """
    #assert X.dtype == W.dtype
    height, width, _ = W.shape
    max_samples, _ = X.shape
    sq_distances = np.empty(shape=(height, width, max_samples), dtype=X.dtype)
    for i in range(height):
        for j in range(width):
            diff = X - W[i,j,:]
            sq_distances[i,j,:] = batch_dot(diff, diff)
    return sq_distances


def avg_distortion(X, W, rate=None):
    """Compute a full-dataset or a stochastic expectation of the distortion,
    that is, the distance between a sample x and its reconstruction
    W[c(x),:].

    """
    height, width, max_features = W.shape
    max_samples, _ = X.shape
    # If a rate is provided, sample the dataset at random at such rate
    if rate != None:
        indices = np.random.randint(0, max_samples, size=int(rate*max_samples))
        X = X[indices,:]
    # Compute winning-neuron (BMU) indices for every input
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    # Compute prototype reconstructions for each input
    reconstructions = W.reshape((-1, max_features))[winning_neurons,:]
    diff = X - reconstructions
    # Avg distortion as mean of squared distances of inputs and BMU prototypes
    reconstruction_errors = batch_dot(diff,diff)
    return np.mean(reconstruction_errors)


#############
#  Testing  #
#############


def SOM_utils_run_tests():
    """Test suite---mainly timings of the distance function variants.
    """
    X = np.random.randn(1000, 100)
    W = np.random.randn(50, 50, 100)
    print('sq_distances_v %.6f sec' % test_timer(lambda: sq_distances_v(X, W), trials=15))
    print('sq_distances_m %.6f sec' % test_timer(lambda: sq_distances_m(X, W), trials=15))
    print('sq_distances_elem %.6f sec' % test_timer(lambda: sq_distances_elem(X, W), trials=15))
    print('compute_sq_distances_v %.6f sec'%
          test_timer(lambda: compute_sq_distances_v(W, X[0]), trials=30))
    print('compute_sq_distances_m %.6f sec' %
          test_timer(lambda: compute_sq_distances_m(W, X[0]), trials=30))
    print('compute_sq_distances_elem %.6f sec' %
          test_timer(lambda: compute_sq_distances_elem(W, X[0]), trials=30))
    

if __name__ == '__main__': SOM_utils_run_tests()
