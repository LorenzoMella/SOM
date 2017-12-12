#===========================================#
#        Principal Component Analysis       #
#                                           #
#        Author: Lorenzo Mella              #
#        Version: 2017-11-27                #
#    Tested on: Python 3.6.3/numpy 1.13.3   #
#===========================================#


import numpy as np
from matplotlib import pyplot
from time import process_time
#np.random.seed(1)


def covariance_eig(X, is_centered=False):
    """ Return eigenvalues and a choice of associated length 1 eigenvectors
    of the sample covariance of the data, ordered by increasing modulus
    
    Parameters
    ----------
    X : ndarray
        the data "design matrix". Each row lists the features of a data-point
    
    is_centered : bool
        if True, no centering is performed on the rows of X
    Returns
    -------
    tuple
        list of eigenvalues and a matrix whose *columns*
        are the associated eigenvectors (essentially the matrix of change of
        coordinates to a system where the cov-matrix of X is diagonal)
    """
    max_samples, max_features = X.shape
    centered_X = X if is_centered else X - np.mean(X, axis=0)
    # DIVIDING MIGHT BE USELESS HERE?
    data_covariance = np.dot(centered_X.T, centered_X) / max_samples
    return np.linalg.eigh(data_covariance)


def principal_components(X, dim=2, screeplot=False):
    # Should be made compatible with negative indices...
    assert dim > 0
    dim = min(dim, X.shape[1])
    X_mean = np.mean(X, axis=0)
    centered_X = X - X_mean
    eigs, eigvs = covariance_eig(centered_X, is_centered=True)
    # Order the eigenvectors from greatest to smallest eigenvalue modulus
    eigvs = eigvs[:, np.flip(np.argsort(np.absolute(eigs)), axis=0)]
    if screeplot:
        pyplot.figure('Eigenvalue Screeplot')
        pyplot.plot(np.flip(np.sort(np.absolute(eigs)), axis=0))
        pyplot.show()
    return np.dot(centered_X, eigvs[:,:dim]), eigvs


def experiment1():
    fan_in = 2
    max_samples = 100000
    
    sigma1 = 1
    sigma2 = 1
    rho = .5
    
    """
    This is A in theory, but we really need the inverse, which
    is easy to compute in closed form
    A = np.array([[sigma1**2, rho*sigma1*sigma2],
                  [rho*sigma1*sigma2, sigma2**2]])
    """
    det_A = (sigma1*sigma2)**2 * (1 - rho**2)
    A_inv = np.array([[sigma2**2, -rho*sigma1*sigma2],
                      [-rho*sigma1*sigma2, sigma1**2]]) / det_A
    # Sufficiently high-variance mean position for the data
    X_mean = 20*np.random.randn(fan_in)
    # Dataset as skewed Gaussian-distributed points
    X = X_mean + np.dot(np.random.randn(max_samples, fan_in), A_inv)
    
    PCA, eigvs = principal_components(X)
    
    print('Sizes of the PCA array: %s' % (PCA.shape,))
    
    # Now we reconstruct X with the eigenvectors
    # and the principal components
    X_mean = np.mean(X, axis=0)
    X_reconstructed = X_mean + np.dot(PCA, eigvs.T)
    
    d = 10
    
    print( 'Good reconstruction! (to %d decimal places)' % (d,)
           if np.all(np.absolute(X - X_reconstructed) < 10**-d)
           else 'Poor reconstruction. (to %d decimal places)' % (d,) )


def experiment2():
    max_fan_in = 28*28+3    # Doesn't make much sense
    max_samples = 60000
    for fan_in in range(2, max_fan_in, 50):
        X = np.random.randn(max_samples, fan_in)
        start = process_time()
        PCA, _ = principal_components(X, fan_in)
        finish = process_time()
        print('PCA time at dimensionality %d = %f s' % (fan_in, finish-start))


def experiment3():
    """Plot PCA reconstruction of a bunch of MNIST images
    """
    from SOM_data_providers import mnist_dataset
    X = mnist_dataset('../../PhD_Datasets/MNIST/train-images-idx3-ubyte')
    X_mean = np.mean(X, axis=0)
    Y, U = principal_components(X, X.shape[1])
    rand_idx = np.random.randint(X.shape[0])
    print('Example chosen at random: %d' % (rand_idx,))
    # Draw pictures of the original and 9 progressively more precise
    # reconstructions
    pyplot.figure('reconstruction')
    pyplot.subplot(251, title='Original')
    pyplot.imshow(X[rand_idx].reshape((28,28)))
    pyplot.subplot(252, title='Data avg.')
    pyplot.imshow(X_mean.reshape((28,28)))
    for max_comps, subplot_num in zip([2,5,10,20,50,100,200,500], range(3,11)):
        # Padding with zeros to embed the approximation into the original
        # dimensionality (784) (components along the eigenvectors)
        y_approx = np.hstack( (Y[rand_idx,:max_comps],
                               np.zeros(X.shape[1]-max_comps)) )
        # Rotate into the original data coordinates and translate to the
        # original center of the cluster (X_mean)
        reconstruction = X_mean + np.dot(y_approx, U.T)
        pyplot.subplot(2,5,subplot_num, title='%d p. comps.' % (max_comps,))
        pyplot.imshow(reconstruction.reshape((28,28)))
    pyplot.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or sys.argv[1] not in ['1', '2', '3']:
        print('Usage: add numerical argument\n'
              '1 - Quality of reconstruction on random 2D points\n'
              '2 - Timings as dimensionality increases\n'
              '3 - MNIST reconstruction increasing principal components')
    elif sys.argv[1] == '1':
        experiment1()
    elif sys.argv[1] == '2':
        experiment2()
    elif sys.argv[1] == '3':
        experiment3()
    
