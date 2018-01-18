#===========================================#
#        SOM - Batch Map version            #
#                                           #
#        Author: Lorenzo Mella              #
#===========================================#


import numpy as np
from time import process_time

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
#from PCA import covariance_eig
from SOM import batch_dot, umatrix, plot_data_and_prototypes
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
    diff = X - W[..., np.newaxis, :]
    return np.sum(diff**2, axis=-1)


def sq_distances_m(X, W):
    """ For some reason this is considerably faster than the v version...
    """
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
    
    sq_dist_m = sq_distances_m(X,W)
    sq_dist_v = sq_distances_v(X,W)
    sq_dist_equal = np.all( np.isclose(sq_dist_m, sq_dist_v,
                                       atol=10**-decimal_places) )
    vor_v = voronoi_cells(X, W)
    vor_m = voronoi_cells(X, W)
    vor_equal = np.all( np.equal(vor_m, vor_v) )
    
    W1 = W
    W2 = np.copy(W)
    
    W1_new = update_W(X, W1)
    W2_new = update_W_e(X, W2)
    
    update_equal = np.all( np.equal(W1_new, W2.new) )
    
    return sq_dist_equal, vor_equal, update_equal


def voronoi_cells(X, W):
    """ Return an array of shape (height, width, max_samples)
        whose boolean entry [i,j,n] is True iff X[n,:] belongs to the Voronoi
        Cell of prototype W[i,j,:].
    """
    max_samples, _ = X.shape
    height, width, _ = W.shape
    # Due to reshaping, this is a matrix whose [p,n] entry is the sq_distance
    # between p-th prototype (linearized order) and the n-th datapoint
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
    prototype_sample_dists = sq_distances_m(X, W).reshape((-1, max_samples))
    closest_prototype = np.argmin(prototype_sample_dists, axis=0)
    num_prototypes, _ = prototype_sample_dists.shape
    mask = np.empty((num_prototypes, max_samples))
    for n in range(max_samples):
        for p in range(height, width):
            mask[p,n] = closest_prototype[n] == p
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


def update_W(X, W):
    mask = voronoi_cells(X, W)
    cell_num_elems = np.sum(mask, axis=-1)
    # Neighborhoods are unions of adjacent cells. We are
    # computing them using a toroidal topology, because it's easier and
    # empirically shown as more meningful on the output-space borders
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
    # neigh_mean_X
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
    print(cell_num_elems)
    
    #print('neigh_not_empty.shape = %s' % (neigh_not_empty.shape,))
    #print('neigh_not_empty all True? %s' % (np.all(neigh_not_empty == True),))
    #print('neigh_sum_X[neigh_not_empty].shape = %s' % (neigh_sum_X[neigh_not_empty].shape,))
    #W[neigh_not_empty,:] = ( neigh_sum_X[neigh_not_empty]
                           #/ neigh_num_elems[neigh_not_empty][:,:, np.newaxis] )
    return W


def update_W_e(X, W):
    height, width, max_features = W.shape
    mask = voronoi_cells_e(X, W)
    cell_num_elems = np.pad( np.sum(mask, axis=-1), pad_width=1, mode='wrap' )
    cell_sum_X = np.pad( sum_cell_e(X, mask), pad_width=((1,1),(1,1),(0,0)),
                        mode='wrap' )
    cell_mean_X = np.empty((height, width, max_features))
#     for i in range(height):
#         for j in range(width):
#             cell_mean_X[i,j,:] = cell_sum_X[i,j,:] / cell_num_elems[i,j]
    # Neighborhoods are unions of adjacent (non diagonal) cells. We are
    # computing them with a toroidal topology, because it's easier and
    # empirically shown as more convenient
    neigh_num_elems = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            neigh_num_elems[i,j] = ( cell_num_elems[i,j]
                                     + cell_num_elems[i,j+1]
                                     + cell_num_elems[i,j+2]
                                     + cell_num_elems[i+1,j]
                                     + cell_num_elems[i+1,j+1]
                                     + cell_num_elems[i+1,j+2]
                                     + cell_num_elems[i+2,j]
                                     + cell_num_elems[i+2,j+1]
                                     + cell_num_elems[i+2,j+2] )
    neigh_sum_X = np.zeros((height, width, max_features))
    for i in range(height):
        for j in range(width):
            neigh_sum_X[i,j,:] = ( cell_sum_X[i,j,:]
                                   + cell_sum_X[i,j+1,:]
                                   + cell_sum_X[i,j+2,:]
                                   + cell_sum_X[i+1,j,:]
                                   + cell_sum_X[i+1,j+1,:]
                                   + cell_sum_X[i+1,j+2,:]
                                   + cell_sum_X[i+2,j,:]
                                   + cell_sum_X[i+2,j+1,:]
                                   + cell_sum_X[i+2,j+2,:] )
    # Update weights
    W_new = np.empty(W.shape)
    for i in range(height):
        for j in range(width):
            if neigh_num_elems[i,j] != 0:
                W_new[i,j,:] = neigh_sum_X[i,j,:] / neigh_num_elems[i,j]
    return W_new


if __name__ == '__main__':
    X, labels, _ = dp.polygon_clusters_dataset()
    indices = np.random.randint(0, X.shape[0], size=height * width)
    W = X[indices,:].reshape((height, width, -1))
    
    for t in range(15):
        W = update_W(X, W)
        
    pyplot.figure('U-Matrix and Input Space Scenario')
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')
    
    plot_data_and_prototypes(X, W)
    pyplot.show()
