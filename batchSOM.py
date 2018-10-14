#################################
#                               #
#    SOM - Batch Map version    #
#                               #
#    Author: Lorenzo Mella      #
#                               #
#################################


# Library imports
import numpy as np
from time import process_time
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Project imports
from PCA import covariance_eig
from SOM_utils import *
from SOM_draw import umatrix, pmatrix, ustarmatrix, plot_data_and_prototypes
import SOM_data_providers as dp
from batchSOM_test import *


# Find all the distances between prototypes and data-points
# The end-result is another 3-array. The first two indices yield the neuron
# position. The third index spans the distances from said neuron to all
# datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
# (height, width, fan_in)
# np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])


def voronoi_cells(X, W):
    """ Return an array of shape (height, width, max_samples)
        whose boolean entry [i,j,n] is True iff X[n,:] belongs to the Voronoi
        Cell of prototype W[i,j,:].
    """
    max_samples, _ = X.shape
    height, width, _ = W.shape
    # Build a matrix whose [p,n] entry is the sq_distance between p-th
    # prototype (linearized order) and the n-th datapoint
    prototype_sample_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    num_prototypes, _ = prototype_sample_dists.shape
    # For each n find the index p of the closest prototype to X[n,:]
    closest_prototype = np.argmin(prototype_sample_dists, axis=0)
    # Convert each entry of closest_prototypes into an indicator vector of the
    # correct index
    indexes = np.arange(num_prototypes)
    mask = np.equal(closest_prototype, indexes[:,np.newaxis])
    return mask.reshape(height, width, max_samples)


def sum_cell(X, mask):
    """ Return an array of shape (height, width, fan_in) whose slice [i,j,:]
        is the vector sum of examples X[n,:] that belong to the Voronoi Cell
        of prototype W[i,j,:].
    """
    return np.dot(mask, X)


def update_W_indicators_vc(X, W, sigma2=16.0):
    self = update_W_indicators_vc
    height, width, max_features = W.shape
    if not hasattr(self, 'D2'):
        """
        Create (and store as property) a lattice and compute square distances
        between any two points of the lattice. The final result is a 4-index
        array, to be intended like so:
            D2[i1,j1,i2,j2] == (i1 - i2)**2 + (j1 - j2)**2
        """
        ii, jj = np.ogrid[:height, :width]
        self.D2 = (  (ii[...,np.newaxis,np.newaxis] - ii)**2
                   + (jj[...,np.newaxis,np.newaxis] - jj)**2 )
    mask = voronoi_cells(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    cell_sum_X = np.dot(mask, X)
    h = self.D2 <= sigma2
    weighted_sum_X = np.dot(h.reshape((height,width,-1)),
                            cell_sum_X.reshape((-1,max_features)))
    weight_sum = np.dot(h.reshape((height,width,-1)),
                        cell_num_elems.reshape((-1,)))
    # Update weights
    W_new = np.divide( weighted_sum_X, weight_sum[...,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    if np.any(bad_indices): print('Possible overflow or division by zero')
    W_new[bad_indices] = W[bad_indices]
    return W_new


def update_W_smooth_vc(X, W, sigma2=16.0):
    self = update_W_smooth_vc
    height, width, max_features = W.shape
    if not hasattr(self, 'D2'):
        """
        Create (and store as property) a lattice and compute square distances
        between any two points of the lattice. The final result is a 4-index
        array, to be intended like so:
            D2[i1,j1,i2,j2] == (i1 - i2)**2 + (j1 - j2)**2
        """
        ii, jj = np.ogrid[:height, :width]
        self.D2 = (  (ii[...,np.newaxis,np.newaxis] - ii)**2
                   + (jj[...,np.newaxis,np.newaxis] - jj)**2 )
    mask = voronoi_cells(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    cell_sum_X = np.dot(mask, X)
    h = np.exp(-0.5 * self.D2 / sigma2)
    weighted_sum_X = np.dot(h.reshape((height,width,-1)),
                            cell_sum_X.reshape((-1,max_features)))
    weight_sum = np.dot(h.reshape((height,width,-1)),
                        cell_num_elems.reshape((-1,)))
    # Update weights
    W_new = np.divide( weighted_sum_X, weight_sum[...,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    if np.any(bad_indices): print('Possible overflow or division by zero')
    W_new[bad_indices] = W[bad_indices]
    return W_new


def update_W_indicators_vc_e(X, W, sigma2=4.0):
    """ In this context, sigma2 is the hard radius of the neighborhood function
    """
    self = update_W_indicators_vc_e
    height, width, max_features = W.shape
    if not hasattr(self, 'ii'):
        self.ii, self.jj = np.ogrid[:height, :width]
    # Compute matrix whose rows are Voronoi Cell indicators
    cell_mask = voronoi_cells_e(X, W)
    # Compute cardinalities of Voronoi Cells
    cell_cardinality = np.sum(cell_mask, axis=-1)
    # Vector sums of datapoints in each Voronoi Cell
    cell_sum_X = sum_cell_e(X, cell_mask)
    # Aggregate the cardinalities of Cells into cardinalities of neighborhoods
    neigh_cardinality = np.zeros((height, width))
    neigh_sum_X = np.zeros((height, width, max_features))
    for i in range(height):
        for j in range(width):
            # Create a boolean matrix: True iff in a spherical neighborhood
            # of the neuron [i,j]
            neigh_mask = (self.ii - i)**2 + (self.jj - j)**2 <= sigma2
            # Sum selected cardinalities
            neigh_cardinality[i,j] = np.sum(neigh_mask * cell_cardinality)
            neigh_sum_X[i,j,:] = np.dot(neigh_mask.reshape((-1,)),
                                        cell_sum_X.reshape((-1, max_features)))
    # Update weights (with check for division by zero, in which case the
    # prototype is not updated)
    W_new = np.divide( neigh_sum_X, neigh_cardinality[...,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    if np.any(bad_indices): print('Possible overflow or division by zero')
    W_new[bad_indices] = W[bad_indices]
    return W_new


def update_W_indicators(X, W, sigma2=16.0):
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    weighted_sum_X = np.zeros((height, width, max_features))
    weight_sum = np.zeros((height, width))
    # Compute the whole neighborhood function for all winning neurons and
    # all neurons under consideration
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    i_win, j_win = np.unravel_index(winning_neurons, dims=(height, width))
    ii, jj = np.ogrid[:height, :width]
    output_sq_dist = (  (ii[..., np.newaxis] - i_win)**2
                      + (jj[..., np.newaxis] - j_win)**2 )
    h = (output_sq_dist <= sigma2)
    # "Matrix" multiplication between h and X weigths the datapoints with their
    # respective neighborhood importance
    weighted_sum_X = np.dot(h, X)
    # Just sum the weights themsemves to get the normalization constant
    weight_sum = np.sum(h, axis=-1)
    # Tacitly assuming that the denominator is never zero...
    W_new = np.divide( weighted_sum_X, weight_sum[:,:,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    if np.any(bad_indices): print('Possible overflow or division by zero')
    W_new[bad_indices] = W[bad_indices]
    return W_new


def update_W_smooth(X, W, sigma2=16.0):
    from scipy.stats import t
    self = update_W_smooth
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    if not hasattr(self, 'ii'):
        self.ii, self.jj = np.ogrid[:height, :width]
    weighted_sum_X = np.zeros((height, width, max_features))
    weight_sum = np.zeros((height, width))
    # Compute the whole neighborhood function for all winning neurons and
    # all neurons under consideration
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    i_win, j_win = np.unravel_index(winning_neurons, dims=(height, width))
    # This is what is computed here:
    #   output_sq_dist[i,j,n] = sq-distance of (i,j) from the winning neuron
    #                           coordinates, relative to input n
    output_sq_dist = (  (self.ii[..., np.newaxis] - i_win)**2
                      + (self.jj[..., np.newaxis] - j_win)**2 )
    h = np.exp( -0.5 * output_sq_dist / sigma2 )
    # Other choices: Cauchy and Student's t pdfs as neighborhood functions
    #h = 1. / (1. + (output_sq_dist / sigma2)**2)
    #h = t(sigma2).pdf(np.sqrt(output_sq_dist))
    """
    We are compactly computing:

         sum( h(j,c(X[n,:])) * X[n,:] for n in range(max_samples) )
        ------------------------------------------------------------
             sum( h(j,c(X[n,:])) for n in range(max_samples) )

    where c(X[n,:]) is the index of the winning neuron of datapoint X[n,:].
    """
    # "Matrix" multiplication between h and X weigths the datapoints with their
    # respective neighborhood importance
    weighted_sum_X = np.dot(h, X)
    # Just sum the weights themsemves to get the normalization constant
    weight_sum = np.sum(h, axis=-1)
    # Protection against division by zero
    W_new = np.divide( weighted_sum_X, weight_sum[:,:,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    if np.any(bad_indices): print('Possible overflow or division by zero')
    W_new[bad_indices] = W[bad_indices]
    return W_new


def W_PCA_initialization(X, shape=(30, 30)):
    height, width = shape
    ii, jj = np.ogrid[:height, :width]
    eigs, eigvs = covariance_eig(X, is_centered=False)
    # We arrange them according to decreasing eigenvalue moduli (hopefully,
    # this is done without rewriting data!)
    eigs = eigs[np.flip(np.argsort(np.absolute(eigs)), axis=0)]
    eigvs = eigvs[:, np.flip(np.argsort(np.absolute(eigs)), axis=0)]
    # Compute the half-lengths of the grid and the spanning unit-vectors
    stds = np.sqrt(eigs[:2])
    new_basis = eigvs[:,:2]
    # Sides of the lattice building block
    unit_y = 2 * stds[0] / (height-1)
    unit_x = 2 * stds[1] / (width-1)
    # Mesh construction
    W = (  (ii*unit_y - stds[0])[...,np.newaxis] * new_basis[:,0]
         + (jj*unit_x - stds[1])[...,np.newaxis] * new_basis[:,1] )
    return W


def get_arguments():
    """ Use the ArgumentParser module to deal flexibly with command-line
        options.
    """
    import argparse
    optparser = argparse.ArgumentParser(description='Self-Organizing Maps - '
                                                    'Batch Algorithm Version.')
    optparser.add_argument('-s', '--size', nargs=2, type=int, default=[40,40],
                           help='height and width of the map')
    optparser.add_argument('-t', '--timesteps', type=int, default=20,
                           help='number of iterations')
    optparser.add_argument('-m', '--minibatch', type=float,
                           help='size of minibatches (% of dataset)')
    optparser.add_argument('-i', '--initialization', type=str,
                           choices=('random', 'data', 'PCA'), default='random',
                           help='type of prototype initialisation')
    optparser.add_argument('-a', '--algorithm', type=str, default='smooth',
                           choices=('smooth', 'smooth_e', 'smooth_vc',
                                    'indicators', 'indicators_vc',
                                    'indicators_vc_e'),
                                    help='height and width of the map')
    optparser.add_argument('-d', '--dataset', type=str, default='polygon',
                           choices=('polygon','rings','iris','irisPCA','mnist',
                                    'mnistPCA'),
                           help='dataset to be analysed')
    # This is now just for the computation of the average distortion
    optparser.add_argument('-v', '--verbose', action='store_true',
                           dest='verbose', default=False,
                           help='visualise additional information (e.g., avg '
                                'distortion')
    return optparser.parse_args()


if __name__ == '__main__':
    # Random seed for testing purposes
    np.random.seed(0)

    algorithms = { 'smooth':          update_W_smooth,
                   'smooth_e':        update_W_smooth_e,
                   'smooth_vc':       update_W_smooth_vc,
                   'indicators':      update_W_indicators,
                   'indicators_vc':   update_W_indicators_vc,
                   'indicators_vc_e': update_W_indicators_vc_e }

    datasets = { 'polygon':  dp.polygon_clusters_dataset,
                 'rings':    dp.linked_rings_dataset,
                 'iris':     dp.iris_dataset,
                 'irisPCA':  dp.iris_dataset_PCA,
                 'mnist':    dp.mnist_dataset,
                 'mnistPCA': lambda: dp.mnist_dataset_PCA(dim=100) }

    args = get_arguments()
    # Self-Organizing Map row and column numbers
    height, width = args.size
    # Number of iterations of batch algorithm
    T = args.timesteps
    # Choice of weight update algorithm
    update_W = algorithms[args.algorithm]
    # Progressively decreasing output-space neighborhood function square-width
    sigma2_i = (0.5 * max(height, width)) ** 2
    sigma2_f = 4.0
    sigma2 = lambda t, T: sigma2_i * (sigma2_f / sigma2_i)**(t / T)
    # Dataset initialization
    X, labels, _ = datasets[args.dataset]()
    max_samples, max_features = X.shape

    # Initialization of the prototypes spherically at random
    if args.initialization == 'data':
        # As a random choice of datapoints
        indices = np.random.randint(0, max_samples, size=height * width)
        W = np.copy(X[indices,:].reshape((height, width, -1)))
    elif args.initialization == 'PCA':
        # As a regular flat sheet oriented with the first 2 principal
        # components of the data
        W = W_PCA_initialization(X, shape=(height, width))
    else:
        # Random initialization
        W = np.random.randn(height, width, max_features)


    # Simulation
    print('Dataset size: %d. Dimensionality: %d\nMap shape: (%d, %d)\n'
          % (max_samples, max_features, height, width))
    for t in range(T):
        start = process_time()
        # In case a minibatch rate is specified, extract a random such sample
        # of the input data
        if args.minibatch and args.minibatch > 0. and args.minibatch <= 1.:
            batch_indices = np.random.randint(max_samples,
                                         size=int(args.minibatch * max_samples))
            X_train = X[batch_indices,:]
        else:
            X_train = X
        W = update_W(X_train, W, sigma2=sigma2(t,T))
        finish = process_time()
        if args.verbose:
            print('Iteration: %d. Update time: %.4f sec. Average distortion: '
                  '%.4f' % (t, finish-start, avg_distortion(X,W)))
        else:
            print('Iteration: %d. Update time: %.4f sec.' % (t, finish-start))

    pyplot.figure('U-Matrix')
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')

    pyplot.figure('P-Matrix')
    pyplot.imshow(pmatrix(X, W))
    pyplot.colorbar()
    pyplot.set_cmap('Greens')

    pyplot.figure('U*-Matrix')
    pyplot.imshow(ustarmatrix(X, W))
    pyplot.colorbar()
    pyplot.set_cmap('ocean')

    plot_data_and_prototypes(X, W, draw_data=True)
    pyplot.show()
