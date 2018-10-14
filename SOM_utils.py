#################################################
#                                               #
#    Utility functions and global parameters    #
#                                               #
#    Author: Lorenzo Mella                      #
#                                               #
#################################################


import numpy as np

# SPECIAL PYTORCH VERSIONS SHOULD ALSO BE INCLUDED

def batch_dot(a, b):
    """Array of dot products over the fastest axis of two arrays. Precisely,
       assuming that a and b have K+1 matching axes,
       batch_dot(a, b)[i1, ..., iK] = a[i1, ..., iK, :].dot(b[i1, ..., iK, :])
    """
    assert a.shape == b.shape
    return np.sum(a*b, axis=-1)

"""
Scores are used to compare the neuronal responses to an input x. Not the best
choice of name, but the neuron with the lowest score wins. Examples of scores
are squared distances (between weights w and x) or inner products w*x. If the
weights are subsequently normalized, there should be no difference between said
kinds of score.
"""
def compute_scores_vec(W, x):
    return batch_dot(W, x)


def compute_sq_distances(W, x):
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


def compute_sq_distances_elem(W, x):
    max_rows, max_cols, fan_in = W.shape
    dists = np.zeros(shape=[max_rows, max_cols])
    for i in range(max_rows):
        for j in range(max_cols):
            diff = W[i,j,:] - x
            dists[i,j] = np.dot(diff, diff)
    return dists


def sq_distances_v(X, W):
    diff = X - W[..., np.newaxis, :]
    return np.sum(diff**2, axis=-1)


def sq_distances_m(X, W):
    """ For some reason this is considerably faster than the v version...
    """
    assert W.dtype == W.dtype
    height, width, _ = W.shape
    max_samples, _ = X.shape
    sq_distances = np.empty(shape=(height, width, max_samples),
                            dtype=X.dtype)
    for i in range(height):
        for j in range(width):
            """
            The content of this scope has been tested as equivalent to the more
            understandable:
            for n in range(max_samples):
                diff = X[n,:] - W[i,j,:]
                sq_distances[i,j,n] = np.dot(diff, diff)
            """
            diff = X - W[i,j,:]
            sq_distances[i,j,:] = batch_dot(diff, diff)
    return sq_distances


def avg_distortion(X, W, rate=None):
    """ Compute a full-dataset or a stochastic expectation of the distortion,
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
