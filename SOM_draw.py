##############################################
#                                            #
#    SOM Visualization Matrices and Tools    #
#                                            #
#    Author: Lorenzo Mella                   #
#                                            #
##############################################


# Library imports
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Project imports
from SOM_utils import compute_sq_distances, avg_distortion


#############
#  Aliases  #
#############


def norm2(vec):
    return np.linalg.norm(vec, ord=2)


def compute_avg_scores(X, W):
    return avg_distortion(X, W, rate=.1)


####################################
#  Cluster Visualization Matrices  #
####################################


def umatrix(W):
    """ Create a U-Matrix (i.e., a map showing how close neighboring units are
        in the feature space)
    """
    height, width, _ = W.shape
    U = np.empty(shape=(height, width))
    for i in range(height):
        for j in range(width):
            # Distances of unit from neighbors on East, South etc.
            # (0 if there is no neighboring unit in that direction)
            de = 0 if j == width-1 else norm2(W[i,j,:] - W[i,j+1,:])
            ds = 0 if i == height-1 else norm2(W[i,j,:] - W[i+1,j,:])
            dw = 0 if j == 0 else norm2(W[i,j,:] - W[i,j-1,:])
            dn = 0 if i == 0 else norm2(W[i,j,:] - W[i-1,j,:])
            # Number of neighbors of a unit: normally 4 but could be 3 or 2
            # if the unit is on the border or a corner
            num_neighs = (i != 0) + (j != 0) + (i != height) + (j != width)
            U[i,j] = (de+ds+dw+dn) / num_neighs
    return U


def pmatrix(X, W):
    """ Create a P-Matrix (i.e., a map showing data density estimation around
        each prototype)
    """
    max_samples, max_features = X.shape
    height, width, _ = W.shape
    # Compute the squared size of the data as 2*variance
    variances = np.var(X, axis=0)
    assert variances.shape == (max_features,)
    # The radius is 20% of the size
    sq_radius = 0.04*(2 * np.max(variances))
    # Compute all prototype-datapoint squared distances
    sq_dists = compute_sq_distances(W[...,np.newaxis,:], X)
    assert sq_dists.shape == (height, width, max_samples)
    # Return the percentage of datapoints within radius (from each prototype)
    return np.mean(sq_dists <= sq_radius, axis=-1)


def ustarmatrix(X, W):
    """ Create a U*-Matrix (i.e., a prototype-distance map modulated by the
        estimated data density)
    """
    pmat = pmatrix(X, W)
    min_pmat = np.min(pmat)
    return umatrix(W) * (pmat - min_pmat) / (np.mean(pmat) - min_pmat)


def mumatrix(X, W):
    raise NotImplementedError


##############
#  Plotting  #
##############


def plot_examples(indices, compute_scores):
    for idx in indices:
        sample_x = X[idx, :]
        sample_label = labels[idx]
        scores = compute_scores(W, sample_x)
        pyplot.figure()
        pyplot.title(str(sample_label))
        pyplot.imshow(scores)
        pyplot.colorbar()
        pyplot.set_cmap('jet')


def plot_data_and_prototypes(X, W, draw_data=True, draw_prototypes=True, axis_on=True):
    """ Plots only the first three components of both the data and the
    SOM prototypes. Also works on 2D data (but drawn flat on a 3D plot)
    """
    height, width, max_features = W.shape
    # Works only for 3D pictures
    assert max_features <= 3
    # Create the third prototype coordinate depending on dimensionality
    Z = np.zeros((height, width)) if max_features == 2 else W[...,2]
    fig = pyplot.figure('Prototypes in the Data Space')
    ax = fig.add_subplot(111, projection='3d')
    if draw_data:
        ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='.', s=25)
    if draw_prototypes:
        ax.plot_wireframe(W[:,:,0], W[:,:,1], Z, linewidth=.3, color='k')
        ax.scatter(W[:,:,0], W[:,:,1], Z, c='b', marker='.', s=100)
    if not axis_on:
        ax.set_axis_off()
