###############################
#                             #
#    Self-Organizing Maps     #
#                             #
#    Author: Lorenzo Mella    #
#                             #
###############################


# Library imports
import time
from sys import version_info
import numpy as np
from matplotlib import pyplot

# Project imports
import SOM_data_providers as dp
from SOM_utils import *
from SOM_draw import *


# Use the new process timers if Python version >= 3.3
py_version, py_subversion = version_info[0:2]
if py_version == 3 and py_subversion >= 3:
    timer = time.process_time
else:
    timer = time.clock


###############################
#  Training Phase Algorithms  #
###############################


def training_phase(X, W, max_steps, phase_name, eta, sigma2, scoring_f):
    raise NotImplementedError
    print('%s.\n----------------------' % (phase_name,))
    for t in range(max_steps):
        start = timer()
        # Draw random input
        sample_x = X[np.random.randint(0,X.shape[0]), :]
        # Compute winning neuron coordinates
        scores = scoring_f(W, sample_x)
        i_win, j_win = winning_neuron(scores, mode=optimizer_mode)
        # Single parameter update
        W = weight_update(W, sample_x, i_win, j_win, eta(t,max_steps),
                          sigma2(t, max_steps))
        finish = timer()
        if t % 500 == 0:
            # Avg squared error refers to sq-distance between data-point and
            # its most responsive ('winning') neuron
            print('Iteration no. %d. Duration (one update): %.3f ms'
                  % (t, 1000.*(finish - start)))
        return W


def weight_update_vec(W, x, i_win, j_win, eta, sigma2):
    """Single weight update for the online SOM algorithm
    """
    # Derive all useful dimensions from W
    height, width, fan_in = W.shape
    assert x.size == fan_in
    # Create the meshgrid if required
    if not hasattr(weight_update_vec, 'mesh_i'):
        weight_update.mesh_i, weight_update.mesh_j = np.ogrid[:height,:width]
    # Compute all neuron squared distances from the winning neuron
    D2 = ( (weight_update_vec.mesh_i - i_win)**2
         + (weight_update_vec.mesh_j - j_win)**2 )
    # Use the squared distances to compute the neighborhood function
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
    W_new = W + eta * h[:,:,np.newaxis] * (x - W)
    # NORMALIZATION HERE PRODUCES INTERESTING RESULTS BUT CONFINES
    # THE PROTOTYPES TO THE PROJECTIVE PLANE (K-1-DIMENSIONAL SPHERE)
    #W_new = W_new / np.sqrt(batch_dot(W_new,W_new))
    return W_new


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


##################
#   Simulation   #
##################


def get_arguments():
    """ Use the ArgumentParser module to deal flexibly with command-line
        options.
    """
    import argparse
    optparser = argparse.ArgumentParser(description='Self-Organizing Maps - '
                                                    'Batch Algorithm Version.')
    optparser.add_argument('-s', '--size', nargs=2, type=int, default=[40,40],
                           help='height and width of the map')
    optparser.add_argument('-t', '--timesteps', nargs=2, type=int,
                           default=[5000,10000], help='number of iterations')
    optparser.add_argument('-i', '--initialization', type=str,
                           choices=('random', 'data', 'PCA'), default='random',
                           help='type of prototype initialisation')
    optparser.add_argument('-d', '--dataset', type=str, default='polygon',
                           choices=('polygon','rings','iris','irisPCA','mnist',
                                    'mnistPCA'),
                           help='dataset to be analysed')
    return optparser.parse_args()


datasets = { 'polygon': dp.polygon_clusters_dataset,
             'rings': dp.linked_rings_dataset,
             'iris':  dp.iris_dataset,
             'irisPCA': dp.iris_dataset_PCA,
             'mnist': dp.mnist_dataset,
             'mnistPCA': lambda: dp.mnist_dataset_PCA(dim=100) }


if __name__ == '__main__':
    # Uncomment for testing purposes
    np.random.seed(123)

    # Retrieve the command-line arguments
    args = get_arguments()

    height, width = args.size
    max_steps_so, max_steps_ft = args.timesteps

    # Initial values of exponentially decreasing parameters
    eta_i = 0.5
    eta_f = 0.01

    sigma2_i = ( 0.5 * max(height, width) )**2 # Squared radius of the 2d array
    sigma2_f = 4.

    # Learning rates
    eta_so = lambda t, T: eta_i * (eta_f / eta_i)**(t / T)
    eta_ft = lambda t, T: eta_f

    # Squared "width" of excitation neighborhoods of the winning neuron
    sigma2_so = lambda t, T: sigma2_i * (sigma2_f / sigma2_i)**(t / T)
    sigma2_ft = lambda t, T: sigma2_f

    compute_scores = compute_sq_distances
    weight_update = weight_update_vec
    optimizer_mode = 'argmin'   # (choose 'argmax' for scalar products)

    # Load the data-file into the "design matrix"
    X, labels, dataset_name = datasets[args.dataset]()

    max_samples, fan_in = X.shape
    print('Number of examples: %d \nNumber of features: %d\n' % X.shape)

    # Initialize network weights WORK IN PROGRESS
    if args.initialization == 'random':
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


#     Save the matrix as a binary file
#     filename = ( 'weights_%s_s%d-%d-%d_i%d-%d%s.npy' % (dataset_name,
#                  height, width, fan_in, max_steps_so, max_steps_ft,
#                  '' if len(sys.argv) < 4 or sys.argv[3] != 'dot' else '_dot') )
#     print('Saving weights as %s' % (filename,))
#     np.save('weights/%s' % (filename,), W)
#     print('... done.')

    # Visualize the scores on some (10) examples
#     indices = np.random.randint(0, max_samples, size=10)
#     plot_examples(indices, compute_scores)

    # Visualize the U-Matrix of the network
    pyplot.imshow(umatrix(W))
    pyplot.colorbar()
    pyplot.set_cmap('plasma')

    plot_data_and_prototypes(X, W)
    pyplot.show()

#     filename = ( 'fig_%s_s%d-%d-%d_i%d-%d%s.pdf' % (dataset_name,
#                  height, width, fan_in, max_steps_so, max_steps_ft,
#                  '' if len(sys.argv) < 4 or sys.argv[3] != 'dot' else '_dot') )
#     pyplot.savefig('figs/%s' % (filename,), format='pdf')
