##################################################
#                                                #
#    SOM - Batch Map - Pytorch Implementation    #
#                                                #
#    Author: Lorenzo Mella                       #
#                                                #
##################################################


import torch

import numpy as np
from time import process_time

from matplotlib import pyplot
from SOM_draw import umatrix, pmatrix, ustarmatrix, plot_data_and_prototypes
import SOM_data_providers as dp
from batchSOM_test import *


#######################
#  Utility Functions  #
#######################


def sq_distances_v(X, W):
    """ Find all the distances between prototypes and data-points
    The end-result is another 3-array. The first two indices yield the neuron
    position. The third index spans the distances from said neuron to all
    datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
    (height, width, fan_in)
    np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])
    """
    diff = X - torch.unsqueeze(W, -2)
    return torch.sum(diff**2, dim=-1)


def batch_dot(a, b):
    assert a.shape == b.shape
    return torch.sum(torch.mul(a, b), dim=-1)


def sq_distances_m(X, W):
    """ For some reason this is considerably faster than the v version...
    """
    height, width, _ = W.shape
    max_samples, _ = X.shape
    sq_distances = torch.empty(size=(height, width, max_samples), dtype=torch.float32)
    for i in range(height):
        for j in range(width):
            """
            The content of this scope has been tested as equivalent to the more
            understandable:
            for n in range(max_samples):
                diff = X[n,:] - W[i,j,:]
                sq_distances[i,j,n] = np.dot(diff, diff)
            """
            diff = X - W[i,j]
            sq_distances[i,j] = batch_dot(diff, diff)
    return sq_distances


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
    closest_prototype = torch.argmin(prototype_sample_dists, dim=0)
    # Convert each entry of closest_prototypes into an indicator vector of the
    # correct index
    indexes = torch.arange(num_prototypes)
    mask = torch.eq(closest_prototype, torch.unsqueeze(indexes, -1))
    return mask.reshape(height, width, max_samples)


def sum_cell(X, mask):
    """ Return an array of shape (height, width, fan_in) whose slice [i,j,:]
        is the vector sum of examples X[n,:] that belong to the Voronoi Cell
        of prototype W[i,j,:].
    """
    return torch.matmul(mask, X)


def clean_bad_indices(W_new, W_old):
    """ Replace any NaN and +/-Inf in W_new with the value of W_old at the same indices.
    """
    bad_indices = torch.isnan(W_new) + W_new.eq(float('inf')) + W_new.eq(float('-inf'))
    if bad_indices.any():
        print('Possible overflow or division by zero')
    W_new[bad_indices] = W_old[bad_indices]
    return W_new


def avg_distortion(X, W, rate=None):
    """ Compute a full-dataset or a stochastic expectation of the distortion,
        that is, the distance between a sample x and its reconstruction
        W[c(x),:].
    """
    height, width, max_features = W.shape
    max_samples, _ = X.shape
    # If a rate is provided, sample the dataset at random at such rate
    if rate != None:
        indices = torch.randint(0, max_samples, size=int(rate*max_samples))
        X = X[indices,:]
    # Compute winning-neuron (BMU) indices for every input
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = torch.argmin(sq_dists, dim=0)
    # Compute prototype reconstructions for each input
    reconstructions = W.reshape((-1, max_features))[winning_neurons,:]
    diff = X - reconstructions
    # Avg distortion as mean of squared distances of inputs and BMU prototypes
    reconstruction_errors = batch_dot(diff,diff)
    return torch.mean(reconstruction_errors)


##############################
#  Weight Update Algorithms  #
##############################


def update_W_indicators_vc(X, W, sigma2=16.0):
    self = update_W_indicators_vc
    height, width, max_features = W.shape
    if not hasattr(self, 'D2'):
        from numpy import ogrid, newaxis
        """
        Create (and store as property) a lattice and compute square distances
        between any two points of the lattice. The final result is a 4-index
        array, to be intended like so:
            D2[i1,j1,i2,j2] == (i1 - i2)**2 + (j1 - j2)**2
        """
        ii, jj = ogrid[:height, :width]
        self.D2 = ((ii[..., newaxis, newaxis] - ii)**2 + (jj[..., newaxis, newaxis] - jj)**2)
        self.D2 = torch.tensor(self.D2, dtype=torch.float32)
    mask = voronoi_cells(X, W)
    cell_num_elems = torch.sum(mask, dim=-1)
    cell_sum_X = torch.dot(mask, X)
    h = torch.le(self.D2, sigma2)
    weighted_sum_X = torch.dot(h.reshape((height, width, -1)), cell_sum_X.reshape((-1, max_features)))
    weight_sum = torch.dot(h.reshape((height, width, -1)), cell_num_elems.reshape((-1,)))
    # Update weights
    W_new = torch.div(weighted_sum_X, torch.unsqueeze(weight_sum, -1))
    return clean_bad_indices(W_new, W)


def update_W_smooth_vc(X, W, sigma2=16.0):
    self = update_W_smooth_vc
    height, width, max_features = W.shape
    if not hasattr(self, 'D2'):
        from numpy import ogrid, newaxis
        """
        Create (and store as property) a lattice and compute square distances
        between any two points of the lattice. The final result is a 4-index
        array, to be intended like so:
            D2[i1, j1, i2, j2] == (i1 - i2)**2 + (j1 - j2)**2
        """
        ii, jj = ogrid[:height, :width]
        self.D2 = ((ii[..., newaxis, newaxis] - ii)**2 + (jj[..., newaxis, newaxis] - jj)**2)
        self.D2 = torch.tensor(self.D2, dtype=torch.float32)
    mask = voronoi_cells(X, W)
    cell_num_elems = torch.sum(mask, dim=-1)
    cell_sum_X = torch.dot(mask, X)
    h = torch.exp(-0.5 * self.D2 / sigma2)
    weighted_sum_X = torch.dot(h.reshape((height,width,-1)), cell_sum_X.reshape((-1,max_features)))
    weight_sum = torch.dot(h.reshape((height,width,-1)), cell_num_elems.reshape((-1,)))
    # Update weights
    W_new = torch.div(weighted_sum_X, torch.unsqueeze(weight_sum, -1))
    return clean_bad_indices(W_new, W)


def update_W_indicators_vc_e(X, W, sigma2=4.0):
    """ In this context, sigma2 is the hard radius of the neighborhood function
    """
    self = update_W_indicators_vc_e
    height, width, max_features = W.shape
    if not hasattr(self, 'ii'):
        from numpy import ogrid
        self.ii, self.jj = ogrid[:height, :width]
        self.ii = torch.tensor(self.ii, dtype=torch.float32)
        self.jj = torch.tensor(self.jj, dtype=torch.float32)
    # Compute matrix whose rows are Voronoi Cell indicators
    cell_mask = voronoi_cells_e(X, W)
    # Compute cardinalities of Voronoi Cells
    cell_cardinality = torch.sum(cell_mask, dim=-1)
    # Vector sums of datapoints in each Voronoi Cell
    cell_sum_X = sum_cell_e(X, cell_mask)
    # Aggregate the cardinalities of Cells into cardinalities of neighborhoods
    neigh_cardinality = torch.zeros(height, width, dtype=torch.long)
    neigh_sum_X = torch.zeros(sizes=(height, width, max_features), dtype=torch.long)
    for i in range(height):
        for j in range(width):
            # Create a boolean matrix: True iff in a spherical neighborhood of the neuron [i,j]
            neigh_mask = (self.ii - i)**2 + (self.jj - j)**2 <= sigma2
            # Sum selected cardinalities
            neigh_cardinality[i,j] = torch.sum(neigh_mask * cell_cardinality)
            neigh_sum_X[i,j,:] = torch.dot(neigh_mask.reshape((-1,)), cell_sum_X.reshape((-1, max_features)))
    # Update weights
    W_new = torch.div(weighted_sum_X, torch.unsqueeze(weight_sum, -1))
    return clean_bad_indices(W_new, W)


def update_W_indicators(X, W, sigma2=16.0):
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    weighted_sum_X = torch.zeros(height, width, max_features)
    weight_sum = torch.zeros(height, width)
    # Compute the whole neighborhood function for all winning neurons and
    # all neurons under consideration
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = torch.argmin(sq_dists, dim=0)
    # Convert winning-neuron indices from flattened representation to 2D coordinates
    i_win = winning_neurons / width
    j_win = winning_neurons % width
    ii, jj = map(lambda arr: torch.tensor(arr, dtype=torch.long), np.ogrid[:height, :width])
    output_sq_dist = (torch.unsqueeze(ii, -1) - i_win)**2 + (torch.unsqueeze(jj, -1) - j_win)**2
    h = output_sq_dist <= sigma2
    # "Matrix" multiplication between h and X weigths the datapoints with their
    # respective neighborhood importance
    weighted_sum_X = torch.dot(h, X)
    # Just sum the weights themsemves to get the normalization constant
    weight_sum = torch.sum(h, dim=-1)
    # Update weights
    W_new = torch.div(weighted_sum_X, torch.unsqueeze(weight_sum, -1))
    return clean_bad_indices(W_new, W)


def update_W_smooth(X, W, sigma2=16.0):
    self = update_W_smooth
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    if not hasattr(self, 'ii'):
        from numpy import ogrid
        self.ii, self.jj = ogrid[:height, :width]
        self.ii = torch.tensor(self.ii, dtype=torch.long)
        self.jj = torch.tensor(self.jj, dtype=torch.long)
    weighted_sum_X = torch.zeros(height, width, max_features)
    weight_sum = torch.zeros(height, width)
    # Compute the whole neighborhood function for all winning neurons
    # and all neurons under consideration
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = torch.argmin(sq_dists, dim=0)
    # Convert winning-neuron indices from flattened representation to 2D coordinates
    i_win = winning_neurons / width
    j_win = winning_neurons % width
    """
    This is what is computed here:
    output_sq_dist[i,j,n] = sq-distance of locations (i,j) on the neuron lattice from each
                            winning neuron location.
    """
    output_sq_dist = torch.tensor((torch.unsqueeze(self.ii, -1) - i_win)**2 +
                                  (torch.unsqueeze(self.jj, -1) - j_win)**2, dtype=torch.float32)
    h = torch.exp(-0.5 * output_sq_dist / sigma2)
    """
    We are compactly computing:

         sum( h(j,c(X[n,:])) * X[n,:] for n in range(max_samples) )
        ------------------------------------------------------------
             sum( h(j,c(X[n,:])) for n in range(max_samples) )

    where c(X[n,:]) is the index of the winning neuron of datapoint X[n,:].
    """
    # "Matrix" multiplication between h and X weigths the datapoints with their
    # respective neighborhood importance
    weighted_sum_X = torch.matmul(h, X)
    # Just sum the weights themsemves to get the normalization constant
    weight_sum = torch.sum(h, dim=-1)
    # Update weights
    W_new = torch.div(weighted_sum_X, torch.unsqueeze(weight_sum, -1))
    return clean_bad_indices(W_new, W)


############################
#  Script and CLI Parsing  #
############################


def get_arguments():
    """ Use the ArgumentParser module to deal flexibly with command-line
        options.
    """
    import argparse
    optparser = argparse.ArgumentParser(description='Self-Organizing Maps - Batch Algorithm Version.')
    optparser.add_argument('-s', '--size', nargs=2, type=int, default=[40,40],
                           help='height and width of the map')
    optparser.add_argument('-t', '--timesteps', type=int, default=20, help='number of iterations')
    optparser.add_argument('-m', '--minibatch', type=float, help='size of minibatches (% of dataset)')
    optparser.add_argument('-i', '--initialization', type=str, choices=('random', 'data'),
                           default='random', help='type of prototype initialisation')
    optparser.add_argument('-a', '--algorithm', type=str, default='smooth',
                           choices=('smooth', 'smooth_e', 'smooth_vc', 'indicators', 'indicators_vc',
                                    'indicators_vc_e'),
                           help='height and width of the map')
    optparser.add_argument('-d', '--dataset', type=str, default='polygon',
                           choices=('polygon', 'rings', 'iris', 'irisPCA', 'mnist', 'mnistPCA'),
                           help='dataset to be analysed')
    # This is now just for the computation of the average distortion
    optparser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False,
                           help='visualise additional information (e.g., avg distortion)')
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
    X = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    max_samples, max_features = X.shape

    # Initialization of the prototypes spherically at random
    if args.initialization == 'data':
        # As a random choice of datapoints
        indices = torch.randint(0, max_samples, size=height * width)
        W = torch.tensor(X[indices,:].reshape((height, width, -1)))
    else:
        # Random initialization
        W = torch.randn(height, width, max_features)

    # Simulation
    print('Dataset size: %d. Dimensionality: %d\nMap shape: (%d, %d)\n'
          % (max_samples, max_features, height, width))
    for t in range(T):
        start = process_time()
        # In case a minibatch rate is specified, extract a random such sample
        # of the input data
        if args.minibatch and args.minibatch > 0. and args.minibatch <= 1.:
            batch_indices = torch.randint(0, max_samples, size=int(args.minibatch * max_samples))
            X_train = X[batch_indices,:]
        else:
            X_train = X
        W = update_W(X_train, W, sigma2=sigma2(t, T))
        finish = process_time()
        if args.verbose:
            print('Iteration: %d. Update time: %.4f sec. Average distortion: %.4f' %
                  (t, finish-start, avg_distortion(X, W)))
        else:
            print('Iteration: %d. Update time: %.4f sec.' % (t, finish-start))

    X = np.array(X)
    W = np.array(W)
    
    pyplot.figure('U-Matrix')
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')

    pyplot.figure('P-Matrix')
    pyplot.imshow(pmatrix(X, W))
    pyplot.colorbar()
    pyplot.set_cmap('Greens')

    plot_data_and_prototypes(X, W, draw_data=True)
    pyplot.show()
