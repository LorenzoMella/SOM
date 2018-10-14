#=====================================#
#        SOM Data Providers           #
#                                     #
#        Author: Lorenzo Mella        #
#=====================================#


import numpy as np
import pandas as ps
from os import environ

dataset_archive = '%s/PhD_Datasets' % (environ['HOME'],)


def malware_dataset(dtype=np.float32):
    dataset_folder = 'malware-benignware-machine-data'
    filename = 'all_malicious_with_headers.txt'
    x_df = ps.read_csv("%s/%s/%s" % (dataset_archive, dataset_folder, filename),
                       delimiter=',', dtype=dtype)
    col_labels = x_df.axes[1]
    # Drop irrelevant features (including class label for now!)
    # and the 14-th faux column
    labels = x_df.as_matrix(columns=(2,))
    x_df = x_df.drop(columns=[col_labels[0],col_labels[2],col_labels[13]])
    # Standardize (order of operations shouldn't make a difference, because
    # std is unaffected by centering. If computations were exact, that is)
    x_df = (x_df - x_df.mean()) / x_df.std()
    # Done with pre-processing, extract raw-data information
    return x_df.values, labels, 'malware'


def iris_dataset(dtype=np.float32):
    dataset_folder = 'iris'
    filename = 'bezdekIris.data'
    x_df = ps.read_csv("%s/%s/%s" % (dataset_archive, dataset_folder, filename),
                       delimiter=',', dtype=dtype)
    # Save and remove labels
    col_labels = x_df.axes[1]
    labels = x_df.as_matrix(columns=[col_labels[-1]])
    print(labels.shape)
    x_df = x_df.drop(columns=[col_labels[-1]])
    # Standardize (order of operations shouldn't make a difference, because
    # std is unaffected by centering. If computations were exact, that is)
    x_df = (x_df - x_df.mean()) / x_df.std()
    # Done with pre-processing, extract raw-data information
    return x_df.values, labels, 'iris'


def iris_dataset_PCA(dtype=np.float32):
    from PCA import principal_components
    raw_values, labels, _ = iris_dataset(dtype=dtype)
    values = principal_components(raw_values, raw_values.shape[1])
    return values, labels, 'iris_PCA'


def polygon_clusters_dataset(std=1):
    """ Generate spherically Gaussian distributed clusters of points at the
        vertices of a cube. The common standard deviation of the Gaussian
        components is specified.
    """
    side = 10.2
    samples_per_cluster = 20
    num_clusters = 8    # 8 vertices of a cube
    max_samples = samples_per_cluster * num_clusters
    X = np.zeros(shape=(max_samples, 3))
    ii, jj, kk = np.meshgrid([-1,1], [-1,1], [-1,1], indexing='ij')
    ii = ii.astype(np.float64) * side
    jj = jj.astype(np.float64) * side
    kk = kk.astype(np.float64) * side
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                idx = 20*(4*i+2*j+k)
                X[idx:idx+20,:] = ( np.array([ii[i,j,k],jj[i,j,k],kk[i,j,k]])
                                + std * np.random.randn(samples_per_cluster,3) )
    # Given the way the data is indexed, generate labels
    labels = np.array([n // samples_per_cluster for n in range(max_samples)])
    return X, labels, 'polygon'


def linked_rings_dataset(std=1, samples_per_point = 10, ring_plot_points=100):
    """ The standard deviation std refers to the one of Gaussian points
        extracted around each ideal point of the circumference
    """
    max_samples_per_ring = samples_per_point * ring_plot_points
    
    radius = 10.2
    angles = np.linspace(0, 2*np.pi, ring_plot_points)
    
    # Horizontal ring
    points_h = np.vstack((np.cos(angles),np.sin(angles),np.zeros_like(angles)))
    points_h = radius * points_h.T - np.array([0.5 * radius, 0, 0])
    offsets = std * np.random.randn(samples_per_point, ring_plot_points, 3)
    X_h = ( points_h + offsets ).reshape(max_samples_per_ring, 3)
    labels_h = np.zeros(shape=(max_samples_per_ring,))
    
    # Vertical ring
    points_v = np.vstack((np.cos(angles), np.zeros_like(angles),np.sin(angles)))
    points_v = radius * points_v.T + np.array([0.5 * radius, 0, 0])
    offsets = std * np.random.randn(samples_per_point, ring_plot_points, 3)
    X_v = ( points_v + offsets ).reshape(max_samples_per_ring, 3)
    labels_v = np.ones(shape=(max_samples_per_ring,))
    
    X = np.vstack( (X_h, X_v) )
    labels = np.hstack( (labels_h, labels_v) )
    
    return X, labels, 'linked_rings'


def mnist_dataset(dtype=np.float32):
    """ Works with all MNIST (vanilla) files (training and test).
        
        Returns:
        -------
        ndarray
    """
    dataset_folder = 'MNIST'
    # Fetch the input data
    filename = 'train-images-idx3-ubyte'
    images_path = '%s/%s/%s' % (dataset_archive, dataset_folder, filename)
    X = extract_idx(images_path, dtype=dtype)
    # The input data is pre-normalized between 0 and 1
    X = X.reshape((X.shape[0], -1)) / 256.
    # Fetch the labels
    filename = 'train-labels-idx1-ubyte'
    labels_path = '%s/%s/%s' % (dataset_archive, dataset_folder, filename)
    labels = extract_idx(labels_path, dtype=np.long)
    return X, labels, 'MNIST'


def extract_idx(path, dtype=np.float32):
    # System endianness: byteorder in ['big', 'little'].
    # The vanilla MNIST files are all big-endian.
    from sys import byteorder
    fp = open(path, 'r')
    # First 2 bytes are zero. The third signals the datatype. The 4th the
    # number of axes of the array
    first_bytes = np.fromfile(fp, dtype=np.uint8, count=4)
    assert first_bytes[0] == 0 and first_bytes[1] == 0
    # THIS IMPLEMENTATION WORKS ONLY IF first_bytes[2] IS 0x8, I.E., IF THE
    # DATA ARE 8-BIT UNSIGNED
    assert first_bytes[2] == 0x08
    ndim = first_bytes[3]
    # The next ndim 32-bit sequences are unsigned integers representing
    # the axes sizes
    sizes = np.fromfile(fp, dtype=np.uint32, count=ndim)
    # Convert if needed
    if byteorder == 'little': sizes.byteswap(True)
    # The first feature represents samples
    max_elements = 1
    for size in sizes: max_elements *= size
    # Luckily, the greyscale levels are expressed as unsigned bytes:
    # no byteswap needed!
    X = np.fromfile(fp, dtype=np.uint8, count=max_elements)
    # Data is converted to the requested data-type, put in table format
    X = X.reshape(sizes).astype(dtype)
    fp.close()
    return X


def mnist_dataset_PCA(dim=None, dtype=np.float32):
    """ Same as mnist_dataset but the data is represented as the first dim
        principal components (all if dim is None or an unreasonable value
        
        Returns:
        -------
        ndarray
    """
    from PCA import principal_components
    X_raw, labels, _ = mnist_dataset(dtype=dtype)
    max_features = X_raw.shape[1]
    # If not specified, use the same dimensionality as the original data
    if dim == None or dim < 0 or dim > max_features:
        dim = max_features
    dataset_name = 'MNIST_PCA%03d' % (dim,)
    X, _ = principal_components(X_raw, dim)
    return X, labels, dataset_name
