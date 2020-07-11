# Util functions in this project

'''
ToDo:
1. fpr/gus selection will select 0 features
2. spearmanr function sometimes cannot calculate results
'''

import json
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import GenericUnivariateSelect, SelectFpr, chi2, mutual_info_classif, f_classif
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.utils import resample

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

# Store the information about regression and test results
def writeLog(x_name, y_name, mean, std, feature_eng, penalty, cv_results, preds, test_results, boot_results):
    '''
    Store a dictionary that contains
    1. x_name: string
    2. y_name: string
    3. mean: [list length = num_features]
    4. std: [list length = num_features]
    5. feature_eng: string
    6. penalty: dict
    7. cv results: dict: list
    8. preds: list
    9. test set results: dict: number
    10. test set bootstrap: dict: list
    '''
    log = {
        'x_name' : x_name,
        'y_name' : y_name,
        'mean' : list(mean),
        'std' : list(std),
        'feature_eng' : feature_eng,
        'penalty' : penalty,
        'cv_results' : cv_results,
        'preds' : list(preds),
        'test_results' : test_results,
        'boot_result' : boot_results
    }
    return log

def saveLog(x_name, y_name, mean, std, feature_eng, penalty, cv_results, preds, test_results, boot_results, verbose = False):
    log = writeLog(x_name, y_name, mean, std, feature_eng, penalty, cv_results, preds, test_results, boot_results)
    file_name = '-'.join([x_name, y_name, feature_eng])
    file_path = os.path.join('log', file_name)
    with open(file_path+'.json', 'w') as f:
        json.dump(log, f)
    if verbose: print('Saved to ',file_path)
    return



# Split train test by csv files in data folder
def splitTrainTest(array, verbose = False):
    test_index_df = loadCsv(['data', 'test_index.csv'])
    test_index_list = list(test_index_df.iloc[:,0])
    test = np.take(array, test_index_list, axis=0)
    train = np.delete(array, test_index_list, axis=0)
    if verbose: print("training set shape: {}; testing set shape: {}".format(train.shape, test.shape))
    return train, test


# Model related
# Feature engineering functions
def featureEng(X, Y, feature_eng, verbose=False):
    '''
    Perform combinations of feature engineering including pca, gus, fpr, chi2
    :param x: input ndarray (n_sample, features)
    :param feature_eng: a string indicating the feature engineering method to use
    Could be:
    '' : no use anything
    'PCA-<explained_variance>'
    'GUS-<function>'
    'FPR-<function>'
    :param verbose: whether to print info
    :return: Processed X, feature_eng model
    '''
    methods_dict = {
        'PCA':pca,
        'GUS':gus,
        'FPR':fpr,
    }
    # if no feature engineering method
    if not feature_eng: return X, 0
    # else
    if verbose: print('Feature Eng with keyword = ', feature_eng)
    method, param = feature_eng.split("-")
    if method == 'PCA' :
        if verbose: print("Using PCA with param = ", param)
        return pca(X, float(param))
    else:
        if verbose: print("Using {} with func = {}".format(method, param))
        return methods_dict.get(method)(X, Y, param)

# PCA
def pca(X, explained_variance = 0.9):
    pcamodel = PCA(n_components=explained_variance)
    pcamodel.fit(X)
    X_selected = pcamodel.transform(X)
    return X_selected, pcamodel

# GUS
def gus(X, Y, func):
    '''
    GenericUnivariateSelect with k_best = 5
    :param func: one of 'f_classif', 'mutual_info_classif', 'chi2'
    :return: transformed x and gusmodel
    '''
    funcs_dict = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'chi2': chi2
    }
    model = GenericUnivariateSelect(funcs_dict.get(func), mode='k_best')
    model.fit(X, Y.reshape(Y.shape[0]))
    X_selected = model.transform(X)
    return X_selected, model

# FPR
def fpr(X,Y,func):
    funcs_dict = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'chi2': chi2
    }
    model = SelectFpr(funcs_dict.get(func))
    model.fit(X, Y.reshape(Y.shape[0]))
    X_selected = model.transform(X)
    return X_selected, model

# perform training and testing on given training and testing set
def modelTrainTest(model, x_train, y_train, x_test, y_test, feature_eng):
    '''
    :return: predictions, [<evaluate returns>]
    '''
    x_train_processed, feature_model = featureEng(x_train, y_train, feature_eng)
    model.fit(x_train_processed, y_train)
    if feature_model: x_test = feature_model.transform(x_test)
    preds = model.predict(x_test)
    return preds, evaluate(y_test, preds)

# Cross validation
def crossValidation(model, X, Y, feature_eng, fold = 5):
    '''
    Perform cross-validation on training set (including standardization, feature engineering)
    :param X: training set X
    :param Y: training set Y
    :param featrue_eng: feature engineering method list to pass in feature engineering function
    :param fold: how many fold, default is 5
    :return: dictionary of metrics, dict['<metric name>'] = [<values, length = fold>]
    '''
    results = []
    kf = KFold(n_splits=fold, random_state=9001, shuffle=True)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        preds, metrics = modelTrainTest(model, x_train, y_train, x_test, y_test, feature_eng)
        results.append(metrics)
    return results

# get a list of named metric
def _getMetricList(results, metric):
    list = []
    for result in results:
        list.append(result[metric])
    return list

# calculate the mean of a metric
def _meanMetric(results, metric):
    return np.mean(_getMetricList(results, metric))

# Hyperparameter tuning function
def outputBestParam(basemodel, param_list, X, Y, feature_eng, metric = 'mae'):
    '''
    :param metric: the one metric to determine best model
    :return: the parameter set of the best model
    '''
    metric_list = []
    for i in range(len(param_list)):
        model = basemodel(**param_list[i])
        results = crossValidation(model, X, Y, feature_eng)
        metric_list.append(_meanMetric(results, metric))
    return param_list[metric_list.index(max(metric_list))]


# bootstrapping
def bootstrap(model, x_train, y_train, x_test, y_test, feature_eng, times = 30):
    '''
    Perform bootstrapping, store results for further analysis and visualization
    :param x_train: training set X
    :param y_train: training set Y
    :param x_test: testing set X
    :param y_test: testing set Y
    :param featrue_eng: feature engineering method list to pass in feature engineering function
    :param times: how many times to bootstrap
    :return: dictionary of metrics, dict['<metric name>'] = [<values, length = fold>]
    '''
    results = []
    index = np.arange(x_train.shape[0])
    for i in range(times):
        boot_index = resample(index, replace=True, n_samples=None, random_state=9001+i)
        x_train_boot, y_train_boot = x_train[boot_index], y_train[boot_index]
        preds, metrics = modelTrainTest(model, x_train_boot, y_train_boot, x_test, y_test, feature_eng)
        results.append(metrics)
    return results

# Print (record) all the metrics needed
def evaluate(y_true, y_preds, verbose = False):
    # mae
    mae = mean_absolute_error(y_true, y_preds)
    rmse = mean_squared_error(y_true, y_preds, squared=False)
    r2 = r2_score(y_true, y_preds)
    cor, p = spearmanr(y_true, y_preds)
    # handle a crash
    if np.isnan(cor): cor = 0
    if np.isnan(p): p = 0
    if verbose:
        print("MAE: ", mae)
        print("RMSE: ", rmse)
        print("R2: ", r2)
        print("Spearman correlation coefficient: {}, p-value = {}".format(cor, p))
    return {'mae' : mae, 'rmse' : rmse, 'r2' : r2, 'cor' :cor ,'p' :p}

# Visualization
# all items in a list with a tab to separate
def returnRow(list):
    return '\t'.join([str(i) for i in list])

# Print results
def fillTable(cv_results, test_results, boot_results):
    row_items = []
    # cv
    for metric in ['mae', 'rmse', 'r2', 'cor']:
        item = _meanMetric(cv_results, metric)
        row_items.append(item)
    # test
    for metric in ['mae', 'rmse', 'r2', 'cor']:
        item = test_results[metric]
        row_items.append(item)
    # boot
    for metric in ['mae', 'rmse', 'r2', 'cor']:
        item = _meanMetric(boot_results, metric)
        row_items.append(item)
    print(returnRow(row_items))
    print('finished')

# debug helper function
def debug():
    a = _getFilePath(['data','X','HM_lin_acc_40.mat'])
    b = loadMatAsArray(a)
    storeNdArray(b, 'HM_lin_acc_40.npy', ['data', 'X'])

#if __name__ == '__main__':
#    debug()