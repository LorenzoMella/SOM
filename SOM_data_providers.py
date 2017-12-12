#===========================================#
#        SOM Data Providers                 #
#                                           #
#        Author: Lorenzo Mella              #
#        Version: 2017-12-01                #
#===========================================#


import numpy as np
import pandas as ps

def malware_dataset():
    import pandas as ps
    
    path = ('/Users/Lorenzo/PhD_Datasets/malware-benignware-machine-data/')
    filename = 'all_malicious_with_headers.txt'
    x_df = ps.read_csv("%s%s" % (path, filename), delimiter=',')
    col_labels = x_df.axes[1]
    # Drop irrelevant features (including class label for now!)
    # and the 14-th faux column
    x_df = x_df.drop(columns=[col_labels[0],col_labels[2],col_labels[13]])
    # Standardize (order of operations shouldn't make a difference, because
    # std is unaffected by centering. If computations were exact, that is)
    x_df = (x_df - x_df.mean()) / x_df.std()
    # Done with pre-processing, extract raw-data information
    return x_df.values


def polygon_clusters_dataset(std=1):
    """ Generate Gaussian distributed clusters of points at the vertices
        of a cube. The standard deviation of the Gaussian is specified.
    """
    side = 10.2
    X = np.zeros(shape=(20*8, 3))
    ii, jj, kk = np.meshgrid([-1,1], [-1,1], [-1,1], indexing='ij')
    ii = ii.astype(np.float64) * side
    jj = jj.astype(np.float64) * side
    kk = kk.astype(np.float64) * side
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                idx = 20*(4*i+2*j+k)
                X[idx:idx+20,:] = ( np.array([ii[i,j,k],jj[i,j,k],kk[i,j,k]])
                                   + std * np.random.randn(20,3) )
    return X


def mnist_dataset(path):
    """ Works with all MNIST (vanilla) files. Training and test, images and
        labels.
        
        Returns:
        -------
        ndarray (dtype=numpy.float64)
    """
    # System endianness: byteorder in ['big', 'little'].
    # The vanilla MNIST files are all big-endian.
    from sys import byteorder
    fp = open(path, 'r')
    # First 2 bytes are zero. The third signals the datatype. The 4th the
    # number of axes of the array
    first_bytes = np.fromfile(fp, dtype=np.uint8, count=4)
    assert first_bytes[0] == 0 and first_bytes[1] == 0
    # File contains greyscale levels in single unsigned bytes
    assert first_bytes[2] == 0x08
    ndim = first_bytes[3]
    # The next ndim 32-bit sequences are unsigned integers representing
    # the axes sizes
    sizes = np.fromfile(fp, dtype=np.uint32, count=ndim)
    # Convert if needed
    if byteorder == 'little': sizes.byteswap(True)
    # Compute the element count in the file (i.e., product of sizes)
    max_samples = sizes[0]
    max_features = 1
    for size in sizes[1:]: max_features *= size
    # Luckily, the greyscale levels are expressed as unsigned bytes:
    # no byteswap needed!
    X = np.fromfile(fp, dtype=np.uint8, count=max_samples*max_features)
    fp.close()
    # Data should be converted to float64, put in table format
    # and normalised between 0 and 1
    return X.reshape((max_samples, max_features)).astype(np.float64) / 256.


def mnist_dataset_PCA(path, dim):
    from PCA import principal_components
    X_raw = mnist_dataset(path)
    return principal_components(X_raw, dim)