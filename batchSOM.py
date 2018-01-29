#===========================================#
#        SOM - Batch Map version            #
#                                           #
#        Author: Lorenzo Mella              #
#===========================================#


import numpy as np
from time import process_time

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from PCA import covariance_eig
from SOM import batch_dot, umatrix, plot_data_and_prototypes
import SOM_data_providers as dp

# Find all the distances between prototypes and data-points
# The end-result is another 3-array. The first two indices yield the neuron
# position. The third index spans the distances from said neuron to all
# datapoints. In other words D[i,j,n] = dist(x[n], w[i,j]).
# (height, width, fan_in)
# np.dot(x[n,:] - w[i,j,:], x[n,:] - w[i,j,:])

def sq_distances_v(X, W):
    diff = X - W[..., np.newaxis, :]
    return np.sum(diff**2, axis=-1)


def sq_distances_m(X, W):
    """ For some reason this is considerably faster than the v version...
    """
    height, width, _ = W.shape
    max_samples, _ = X.shape
    sq_distances = np.empty( shape=(height, width, max_samples),
                             dtype=np.float64 )
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


def sum_cell(X, mask):
    """ Return an array of shape (height, width, fan_in) whose slice [i,j,:]
        is the vector sum of examples X[n,:] that belong to the Voronoi Cell
        of prototype W[i,j,:].
    """
    return np.dot(mask, X)


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


def update_W(X, W, sigma2=16.0):
    mask = voronoi_cells(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    # Neighborhoods are unions of adjacent cells. We are
    # computing them using a toroidal topology, because it's easier and
    # empirically shown as more meningful on the output-space borders
    # DESPITE THIS, A SQUARE TOPOLOGY WITH BARRIERS SHOULD BE IMPLEMENTED WITH
    # HIGHER PRIORITY
    neigh_num_elems = ( cell_num_elems
                        + np.roll(cell_num_elems, shift=-1, axis=0)
                        + np.roll(cell_num_elems, shift=1, axis=0)
                        + np.roll(cell_num_elems, shift=-1, axis=1)
                        + np.roll(cell_num_elems, shift=1, axis=1)
                        + np.roll(cell_num_elems, shift=(-1,-1), axis=(0,1))
                        + np.roll(cell_num_elems, shift=(-1,1), axis=(0,1))
                        + np.roll(cell_num_elems, shift=(1,-1), axis=(0,1))
                        + np.roll(cell_num_elems, shift=(1,1), axis=(0,1)) )
    print('neigh_num_elems.dtype = %s' % (neigh_num_elems.dtype,))
    cell_sum_X = sum_cell(X, mask)
    neigh_sum_X = ( cell_sum_X
                    + np.roll(cell_sum_X, shift=-1, axis=0)
                    + np.roll(cell_sum_X, shift=1, axis=0)
                    + np.roll(cell_sum_X, shift=-1, axis=1)
                    + np.roll(cell_sum_X, shift=1, axis=1)
                    + np.roll(cell_sum_X, shift=(-1,-1), axis=(0,1))
                    + np.roll(cell_sum_X, shift=(-1,1), axis=(0,1))
                    + np.roll(cell_sum_X, shift=(1,-1), axis=(0,1))
                    + np.roll(cell_sum_X, shift=(1,1), axis=(0,1)) )
    # Update weights
    neigh_not_empty = np.nonzero(neigh_num_elems)
    print(np.mean(cell_sum_X, axis=(0,1)))
    print(np.mean(cell_num_elems))
    #print('neigh_not_empty.shape = %s' % (neigh_not_empty.shape,))
    #print('neigh_not_empty all True? %s' % (np.all(neigh_not_empty == True),))
    #print('neigh_sum_X[neigh_not_empty].shape = %s'
    #      % (neigh_sum_X[neigh_not_empty].shape,))
    #W[neigh_not_empty,:] = ( neigh_sum_X[neigh_not_empty]
                           #/ neigh_num_elems[neigh_not_empty][:,:, np.newaxis] )
    # Update weights
    height, width, _ = W.shape
    W_new = np.empty(W.shape)
    """
    A faster alternative could look like this:
            safe_indices = np.nonzero(neigh_num_elems)
            unsafe_indices = np.nonzero(neigh_num_elems == 0)
            W_new[safe_indices,:] = ( neigh_sum_X[safe_indices,:]
                                     / neigh_num_elems[safe_indices] )
            W_new[unsafe_indices,:] = W[unsafe_indices,:]
    """ 
    for i in range(height):
        for j in range(width):
            if neigh_num_elems[i,j] != 0:
                W_new[i,j,:] = neigh_sum_X[i,j,:] / neigh_num_elems[i,j]
            else:
                W_new[i,j,:] = W[i,j,:]
    return W_new


def update_W_e(X, W, sigma2=4.0):
    """ In this context, sigma2 is the hard radius of the neighborhood function
    """
    height, width, max_features = W.shape
    if not hasattr(update_W_e, 'ii'):
        update_W_e.ii, update_W_e.jj = np.ogrid[:height, :width]
    # Compute matrix whose rows are Voronoi Cell indicators
    cell_mask = voronoi_cells_e(X, W)
    # Compute cardinalities of Voronoi Cells
    cell_cardinality = np.sum(cell_mask, axis=-1)
    # Vector sums of datapoints in each Voronoi Cell
    cell_sum_X = sum_cell_e(X, cell_mask)
    # Aggregate the cardinalities of Cells into cardinalities of neighborhoods
    # THIS MUST BE THE SLOW PART BUT I SUSPECT THAT MASKING OPERATIONS ARE
    # EXPENSIVE IN GENERAL. SOMETIMES REPEATED COMPUTATIONS IN A COMPACT BLOCK
    # LEFT TO BLAS ARE EXECUTED FASTER THAN A SEQUENCE OF "CLEVER" PREPROCESSING
    # IDEAS...
    neigh_cardinality = np.zeros((height, width))
    neigh_sum_X = np.zeros((height, width, max_features))
    for i in range(height):
        for j in range(width):
            # Create a boolean matrix: True iff in a spherical neighborhood
            # of the neuron [i,j]
            neigh_mask = (update_W_e.ii-i)**2 + (update_W_e.jj-j)**2 <= sigma2
            # Sum selected cardinalities
            neigh_cardinality[i,j] = np.sum(neigh_mask * cell_cardinality)
            neigh_sum_X[i,j,:] = np.dot(neigh_mask.reshape((-1,)),
                                        cell_sum_X.reshape((-1, max_features)))
    # Update weights (with check for division by zero, in which case the
    # prototype is not updated)
    W_new = np.divide( neigh_sum_X, neigh_cardinality[...,np.newaxis] )
    bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
    W_new[bad_indices] = W[bad_indices]
    return W_new


def update_W_indicators(X, W, sigma2=16.0):
    raise NotImplementedError
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    weighted_sum_X = np.zeros((height, width, max_features))
    weight_sum = np.zeros((height, width))
    # Compute the whole neighborhood function for all winning neurons and
    # all neurons under consideration (4 indices)
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    i_win, j_win = np.unravel_index(winning_neurons, dims=(height, width))
    ii, jj = np.ogrid[:height, :width]
    output_sq_dist = (  (ii[..., np.newaxis] - i_win)**2
                      + (jj[..., np.newaxis] - j_win)**2 )
    h = np.exp( -0.5 * output_sq_dist / sigma2 )
    # "Matrix" multiplication between h and X weigths the datapoints with their
    # respective neighborhood importance
    weighted_sum_X = np.dot(h, X)
    # Just sum the weights themsemves to get the normalization constant
    weight_sum = np.sum(h, axis=-1)
    # Tacitly assuming that the denominator is never zero...
    return weighted_sum_X / weight_sum[:,:,np.newaxis]


def update_W_smooth(X, W, sigma2=16.0):
    max_samples, _ = X.shape
    height, width, max_features = W.shape
    weighted_sum_X = np.zeros((height, width, max_features))
    weight_sum = np.zeros((height, width))
    # Compute the whole neighborhood function for all winning neurons and
    # all neurons under consideration (4 indices)
    sq_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    winning_neurons = np.argmin(sq_dists, axis=0)
    i_win, j_win = np.unravel_index(winning_neurons, dims=(height, width))
    ii, jj = np.ogrid[:height, :width]
    output_sq_dist = (  (ii[..., np.newaxis] - i_win)**2
                      + (jj[..., np.newaxis] - j_win)**2 )
    h = np.exp( -0.5 * output_sq_dist / sigma2 )
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
    # Tacitly assuming that the denominator is never zero...
    return weighted_sum_X / weight_sum[:,:,np.newaxis]


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


def W_PCA_initialization(X, shape=(30, 30)):
    height, width = shape
    ii, jj = np.ogrid[:height, :width]
    eigs, eigvs = covariance_eig(X, is_centered=False)
    # We arrange them according to decreasing eigenvalue moduli
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
    optparser.add_argument('--size', nargs=2, type=int, default=[40,40],
                           help='height and width of the map') 
    optparser.add_argument('--timesteps', type=int, default=20,
                           help='number of iterations')
    optparser.add_argument('--minibatch', type=float,
                           help='size of minibatches (% of dataset)')
    optparser.add_argument('--initialization', type=str,
                           choices=('random', 'data', 'PCA'), default='random',
                           help='type of prototype initialisation')
    return optparser.parse_args()


if __name__ == '__main__':
    np.random.seed(0)
    args = get_arguments()
    # Self-Organizing Map row and column numbers
    height, width = args.size
    # Number of iterations of batch algorithm
    T = args.timesteps
    print(T)
    # Progressively decreasing output-space neighborhood function square-width
    sigma2_i = (0.5 * max(height, width)) ** 2
    sigma2_f = 1.0
    sigma2 = lambda t, T: sigma2_i * (sigma2_f / sigma2_i)**(t / T)
    
    # Dataset initialization
    X, labels, _ = dp.linked_rings_dataset()
    #X, labels, _ = dp.polygon_clusters_dataset()
    #X, labels, _ = dp.mnist_dataset_PCA(dim=100)
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
    
    for t in range(T):
        start = process_time()
        # In case a minibatch rate is specified, extract a random such sample
        # of the input data
        if args.minibatch and args.minibatch > 0. and args.minibatch <= 1.:
            batch_indices = np.random.randint(max_samples,
                                         size=int(args.minibatch * max_samples))
            X_train = X[batch_indices, :]
        else:
            X_train = X
        W = update_W_smooth(X_train, W, sigma2=sigma2(t,T))
        finish = process_time()
        print('Iteration: %d. Update time: %f sec' % (t, finish-start))
    
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')
    
    plot_data_and_prototypes(X, W, draw_data=False)
    pyplot.show()
