# Util functions in this project

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

# Data related

# Output path of a file given a list of names
def _getFilePath(list):
    return os.path.join(*list)

# Load mat file as numpy.ndarray
def loadMatAsArray(path_list):
    file = _getFilePath(path_list)
    mat = loadmat(file)
    key = list(mat.keys())[-1]
    return mat[key]

# load npy file as numpy.ndarray
def loadNpy(path_list):
    file = _getFilePath(path_list)
    return np.load(file)

# Load csv file as numpy.ndarray
def loadCsv(path_list):
    file = _getFilePath(path_list)
    return pd.read_csv(file)


# Store numpy.ndarray to npy file
def storeNdArray(array, name, subdir = []):
    dir = _getFilePath(subdir + [name])
    with open(dir,'wb') as f:
        np.save(f, array)
    print("Saved {} to {}".format(name, dir))
    return

# Split train test by csv files in data folder
def splitTrainTest(array, verbose = False):
    test_index_df = loadCsv(['data', 'test_index.csv'])
    test_index_list = list(test_index_df.iloc[:,0])
    test = np.take(array, test_index_list, axis=0)
    train = np.delete(array, test_index_list, axis=0)
    if verbose: print("training set shape: {}; testing set shape: {}".format(train.shape, test.shape))
    return train, test


# debug helper function
def debug():
    a = _getFilePath(['data','X','HM_lin_acc_40.mat'])
    b = loadMatAsArray(a)
    storeNdArray(b, 'HM_lin_acc_40.npy', ['data', 'X'])


#if __name__ == '__main__':
#    debug()