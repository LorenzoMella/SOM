#===========================================#
#        Self-Organizing Maps               #
#                                           #
#        Author: Lorenzo Mella              #
#        Version: 2017-12-04                #
#    Tested on: Python 2.7.6/numpy 1.13.3   #
#               Python 3.6.3/numpy 1.13.3   #
#===========================================#

# (Current version is just a script with global variables and functions)

import sys  # for command line arguments
import time
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import SOM_data_providers as dp

# Use the new process timers if Python version >= 3.3
py_version, py_subversion = sys.version_info[0:2]
if py_version == 3 and py_subversion >= 3:
    timer = time.process_time
else:
    timer = time.clock

# Uncomment for testing purposes
np.random.seed(123)


############################################
# Utility functions and global parameters  #
############################################

# Lattice size
height = 40
width = 40

# Initial values of exponentially decreasing parameters
eta_i = 0.5
eta_f = 0.01
sigma2_i = ( 0.5 * max(height, width) )**2 # The squared radius of the 2d array
sigma2_f = 9.

# Learning rates
eta_so = lambda t, T: eta_i * (eta_f / eta_i)**(t / T)
eta_ft = lambda t, T: eta_f

# Squared "width" of excitation neighborhoods of the winning neuron
sigma2_so = lambda t, T: sigma2_i * (sigma2_f / sigma2_i)**(t / T)
sigma2_ft = lambda t, T: sigma2_f

# Mesh grid to compute distances
# This precomputation trick makes sense for at most 5D grids.
# The size obviously grows exponentially!
mesh_i, mesh_j = np.meshgrid(range(width), range(height), indexing='ij')

def batch_dot(a, b):
    """Array of dot products over the fastest axis of two arrays. Precisely,
       assuming that a and b have K+1 matching axes,
       batch_dot(a, b)[i1, ..., iK] = a[i1, ..., iK, :].dot(b[i1, ..., iK, :])
    """
    assert a.shape == b.shape
    return np.sum(a*b, axis=-1)


def weight_update_vec(W, x, i_win, j_win, eta, sigma2):
    """Single weight update for the online SOM algorithm
    """
    # Derive all useful dimensions from W
    height, width, fan_in = W.shape
    assert x.size == fan_in
    # Compute all neuron distances from the winning neuron
    D2 = (mesh_i - i_win)**2 + (mesh_j - j_win)**2
    h = np.exp(-0.5 * D2 / sigma2)
    # Update all the weights.
    # An axis must explicitly be added to h for correct broadcasting:
    #
    #   h[:,:,np.newaxis].shape ==                (height, width,      1)
    #   sample_x.shape broadcasts to              (     1,     1, fan_in)
    #   W.shape ==                                (height, width, fan_in)
    #
    # The resulting algorithm is like doing, for all i,j,
    #
    #   W_new[i,j,:] = W_new[i,j,:] = eta*h[i,j]*(sample_x - W[i,j,fan_in])
    W_new = W + eta * h[:,:,np.newaxis] * (sample_x - W)
    # NORMALIZATION HERE PRODUCES INTERESTING RESULTS BUT CONFINES
    # THE PROTOTYPES ON AN K-1-DIM SPHERE
    #W_new = W_new / np.sqrt(batch_dot(W_new,W_new))
    return W_new


"""
Scores are used to compare the neuronal responses to an input x. Not the best
choice of name, but the neuron with the lowest score wins. Examples of scores
are squared distances (between weights w and x) or inner products w*x. If the
weights are subsequently normalized, there should be no difference between said
kinds of score.
"""
def compute_scores_vec(W, x):
    # Broadcasted scalar products of x with each weight vector
    # SEE IF THIS CAN BE REDUCED TO COMMON BROADCASTING
    BX = np.broadcast_to(x, W.shape)
    # Result is a 2d matrix of scalar products
    return batch_dot(W, BX)


def compute_sq_distances(W, x):
    diff = W - x
    return batch_dot(diff, diff)


def winning_neuron(scores, mode='argmin'):
    assert mode in ['argmin', 'argmax']
    find_optimum = np.argmin if mode == 'argmin' else np.argmax
    # Compute the optimizer as a linearized index on the flattened scores array
    flat_optimizer = find_optimum(scores)
    # Restore the neuron identifier in two-index form
    return np.unravel_index(flat_optimizer, dims=scores.shape)


def weight_update_elem(W, x, i_win, j_win, eta, sigma2):
    """More elementary (hence, probably correctly designed and not buggy)
       versions of the above for testing purposes. It still uses numpy for
       scalar products and 2-norms (restricted to single 1D vectors)
    """
    # Derive all useful dimensions from W
    height, width, fan_in = W.shape
    assert x.size == fan_in
    # Allocate new weight array
    W_new = np.empty_like(W, dtype=np.float64)
    # SLOW neuron-by-neuron weight update
    for i in range(height):
        for j in range(width):
            d2 = (i-i_win)**2 + (j-j_win)**2
            h = np.exp(-0.5 * d2 / sigma2)
            W_new[i,j,:] = W[i,j,:] + eta*h*(x - W[i,j,:])
            # W_new[i,j,:] /= np.linalg.norm(W_new[i,j,:], ord=2)
    return W_new


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


#######################################################
#   Clustering and performance assessment functions   #
#######################################################

def umatrix(W):
    """Create and visualize U-Matrix (i.e., a map showing how close neighboring
       units are in the feature space)
    """
    norm2 = lambda vec: np.linalg.norm(vec, ord=2)
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


def compute_avg_scores(W, X):
    # Select some 10% of examples at random from X
    # Compute scores for each (can be most probably done in one stroke...?)
    # Choose winning neuron for each
    # Return avg sq-distance between example x and its winning neuron weights
    raise NotImplementedError


##################
#   Simulation   #
##################

if __name__ == '__main__':
    
    # Number of iterations (THESE RULE-OF-THUMB VALUES WORK, BUT A STOPPING
    # CRITERION IS MORE DESIRABLE. ALSO, WE WANT TO USE A GOOD PORTION OF THE
    # DATA OR, IF REASONABLE, SCANNING THE WHOLE DATASET MULTIPLE TIMES)
    
    # Defaults
    max_steps_so = 5000
    max_steps_ft = 1250000
    compute_scores = compute_sq_distances
    weight_update = weight_update_vec
    optimizer_mode = 'argmin'   # argmax of scalar products
    # Updates of defaults based on command line arguments
    if len(sys.argv) >= 2: max_steps_so = int(sys.argv[1])
    if len(sys.argv) >= 3: max_steps_ft = int(sys.argv[2])
    if len(sys.argv) >= 4:
        assert sys.argv[3] in ['dot_vec', 'dot_elem', 'sq_vec', 'sq_elem']
        if sys.argv[3] == 'dot_elem': compute_scores = compute_scores_elem
        elif sys.argv[3] == 'dot_vec': compute_scores = compute_scores_vec
        elif sys.argv[3] == 'sq_vec': compute_scores = compute_sq_distances
        elif sys.argv[3] == 'sq_elem': compute_scores = compute_sq_distances_elem
        if sys.argv[3] in ['dot_elem', 'sq_elem']:
            weight_update = weight_update_vec
        # argmin of squared distances
        optimizer_mode = 'argmin' if sys.argv[3] in ['sq_vec', 'sq_elem'] \
                          else 'argmax'
    
    # Load the data-file into the "design matrix" x
    X = dp.polygon_clusters_dataset()
    dataset_name = 'polygon'
#     X = dp.mnist_dataset('../../PhD_Datasets/MNIST/train-images-idx3-ubyte')
#     dataset_name = 'mnist'
#     X, _ = dp.mnist_dataset_PCA('../../PhD_Datasets/MNIST/train-images-idx3-ubyte',
#                                 dim=100)
#     dataset_name = 'mnist_pca'
    max_samples, fan_in = X.shape
    print('Number of examples: %d \nNumber of features: %d\n' % X.shape)
    
    # Initialize network weights (A BETTER WAY IS TO USE THE DATAPOINTS THEMSELVES)
    W = np.random.randn(height, width, fan_in)
    
    # Self-organizing phase
    print('Self-organizing phase.\n----------------------')
    for t in range(max_steps_so):
        start = timer()
        # Draw random input
        sample_x = X[np.random.randint(0,max_samples),:]
        # Compute winning neuron coordinates
        scores = compute_scores(W, sample_x)
        i_win, j_win = winning_neuron(scores, mode=optimizer_mode)
        # Single parameter update
        W = weight_update(W, sample_x, i_win, j_win, eta_so(t,max_steps_so),
                          sigma2_so(t, max_steps_so))
        finish = timer()
        if t % 500 == 0:
            # Avg squared error refers to sq-distance between data-point and
            # its most responsive ('winning') neuron
            print('Iteration no. %d. Duration (one update): %.3f ms'
                  % (t, 1000.*(finish - start)))
    
    # Fine-tuning phase
    print('Fine-tuning phase.\n------------------')
    for t in range(max_steps_ft):
        start = timer()
        # Draw random input
        sample_x = X[np.random.randint(0,max_samples),:]
        # Compute winning neuron coordinates
        scores = compute_scores(W, sample_x)
        i_win, j_win = winning_neuron(scores, mode=optimizer_mode)
        # Single parameter update
        W = weight_update(W, sample_x, i_win, j_win, eta_ft(t, max_steps_ft),
                          sigma2_ft(t, max_steps_ft))
        finish = timer()
        if t % 500 == 0:
            print('Iteration no. %d. Duration (one update): %.3f ms'
                  % (t, 1000.*(finish - start)))
    
    # Save the matrix as a binary file
    filename = ( 'weights_%s_s%d-%d-%d_i%d-%d%s.npy' % (dataset_name,
                 height, width, fan_in, max_steps_so, max_steps_ft,
                 '' if len(sys.argv) < 4 or sys.argv[3] != 'dot' else '_dot') )
    print('Saving weights as %s' % (filename,))
    np.save(filename, W)
    print('... done.')
    
    # Visualize the scores on some examples
    for i in range(10):
        sample_x = X[np.random.randint(0,max_samples),:]
        scores = compute_scores(W, sample_x)
        pyplot.matshow(scores)
        pyplot.colorbar()
        pyplot.set_cmap('jet')
    
    # Visualize the U-Matrix of the network
    pyplot.figure('U-Matrix and Input Space Scenario')
    #pyplot.subplot(121, title='U-Matrix')
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')
    
#     fig = pyplot.figure('U-Matrix and Input Space Scenario')
#     ax = fig.add_subplot(122, projection='3d', title='Input Space')
#     W = W.reshape((W.shape[0]*W.shape[1], W.shape[2]))
#     ax.scatter(W[:,0], W[:,1], W[:,2], 'b.')
#     ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='^')
#     ax.set_xlabel('i')
#     ax.set_ylabel('j')
#     ax.set_zlabel('k')
    
    pyplot.show()