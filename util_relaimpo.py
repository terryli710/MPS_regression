# This file contains util functions to calculate relative importance in linear models

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations, compress
from tqdm import tqdm

### COLLINEARITY RELATED ###
def getVif(xdf):
    '''
    # Calculates VIF
    :param X: expected to be a dataframe
    :return: a DF containing "VIF" column to show information
    '''
    vif = pd.DataFrame()
    vif['variable'] = xdf.columns
    vif["VIF"] = [variance_inflation_factor(xdf.values, i) for i in range(xdf.shape[1])]
    return(vif)

def _getOobVif(vif, threshold):
    variables = []
    oob_vifs = []
    for i in range(vif.shape[0]):
        if vif.loc[i,'VIF'] > threshold:
            variables.append(vif.loc[i, 'variable'])
            oob_vifs.append(vif.loc[i, 'VIF'])
    return pd.DataFrame({'variable':variables, 'VIF':oob_vifs})

def _getMaxVif(vif):
    return vif.loc[vif['VIF'].idxmax()]

def vifStepwiseSelect(X, threshold=5, verbose=1):
    '''
    perform stepwise feature selection using VIF to deal with collinearity
    :param X: expecting a dataframe
    :param threshold: maximum vif value of the remaining variables
    :return: selected X; vif dataframe for checking
    '''
    i = 1
    deleted = []
    x_copy = X.copy()
    while True:
        vif = getVif(x_copy)
        oob_vif = _getOobVif(vif, threshold)
        # IF there is something to delete
        if oob_vif.shape[0] > 0:
            max_vif = _getMaxVif(vif)
            variable_delete = str(max_vif['variable'])
            x_copy = x_copy.drop(variable_delete, axis=1)
            deleted.append(variable_delete)
            i += 1
        else:
            if verbose:
                print('Deleted variable index: ', deleted)
                print('{} variables remain in the model'.format(X.shape[1]-len(deleted)))
            return x_copy, vif
        # print info for each iteration
        if verbose >= 2:
            print("Iteration ", i)
            if verbose >= 3:
                print("Problematic variables and vifs are: ")
                print(oob_vif.head())
            if verbose >= 2:
                print("Problematic variables are: ", list(oob_vif.variable))
                print("Delete variable",str(max_vif['variable']))

### FEATURE IMPORTANCE RELATED ###
# reference: https://conservancy.umn.edu/bitstream/handle/11299/213096/Semmel_umn_0130E_18941.pdf?sequence=1&isAllowed=y
# Here I will implement
# Zero-Order Correlation Coefficients (first)
# Standardized Regression Weights (srw)
# Structure Coefficients (structcoef)
# Pratt Measure (pratt)
# Commonality Analysis (ca)


# 1. first
def _r2(y, yhat):
    # calculate r squared
    # r2 = 1 - SSR/SST
    ssr = np.sum(np.square(y-yhat))
    sst = np.sum(np.square(y-np.mean(y)))
    r2 = 1 - float(ssr)/sst
    return r2

def _stdLinear(x, y):
    ss = StandardScaler()
    xs = ss.fit_transform(x)
    lr = LinearRegression()
    lr.fit(xs, y)
    return lr, ss

def first(x, y):
    if not isinstance(x, list):
        # return the R^2 of p linear models
        # model i: y = beta * x_i + intercept
        r2_list = []
        for i in range(x.shape[1]):
            r2_list.append(_getR2(x.iloc[:,i:i+1], y))
        return r2_list
    elif isinstance(x, list):
        # input a list of dataframes
        # return the R^2 of each models
        r2_list = []
        for xdf in x:
            r2_list.append(_getR2(xdf, y))
        return r2_list


# 2. srw
def srw(x, y):
    lr,_ = _stdLinear(x, y)
    return [abs(x) for x in list(lr.coef_[0,:])]

# 3. structcoef
def structcoef(x, y):
    if not isinstance(x, list) : r2_total = _getR2(x, y)
    else : r2_total = _getR2(pd.concat(x, axis=1), y)
    return first(x, y) / r2_total


# 4. pratt
def pratt(x, y):
    beta = [abs(x) for x in srw(x, y)]
    r2 = first(x, y)
    return [a*b for a,b in zip(beta, r2)]

# 5. Commonality Analysis (ca)
def _getR2(X, Y):
    # get R^2 from a simple linear regression between X and Y
    lr = LinearRegression()
    lr.fit(X, Y)
    return _r2(Y, lr.predict(X))


# other utils
def _getRank(array, absolute=True, names = []):
    '''
    Give the rank of a np.array in a list
    Input: array: np.array, can be coefficient for each feature
           names: feature name in a df
           verbose: bool, print ranking or not
    '''
    assert type(array) == np.ndarray
    array = array.reshape(-1, )
    # two ways of ranking
    if absolute: st_idx = np.argsort(-abs(array))
    else : st_idx = np.argsort(-array)
    rank = []
    for i in range(array.shape[0]):
        rank.append(np.where(st_idx == i)[0][0] + 1)
    if len(names)>0: return pd.DataFrame({'names':names, 'rank':rank})
    else: return pd.DataFrame({'names':list(range(len(rank))), 'rank':rank})


def bootstrapping(x, y, func, times=100):
    # version 1
    if isinstance(x, pd.DataFrame):
        coef_boot = np.zeros((times, x.shape[1]))
        for t in tqdm(range(times)):
            # boot sample
            num_sample = x.shape[0]
            idx_boot = np.random.randint(0, num_sample, size=num_sample)
            x_boot = x.iloc[idx_boot, :]
            y_boot = y[idx_boot]
            # get feature importance
            coef_boot[t,:] = func(x_boot, y_boot)
    # version 2
    elif isinstance(x, list):
        coef_boot = np.zeros((times, len(x)))
        num_sample = x[0].shape[0]
        for t in tqdm(range(times)):
            # boot sample
            idx_boot = np.random.randint(0, num_sample, size=num_sample)
            x_boot = [x_item.iloc[idx_boot,:] for x_item in x]
            y_boot = y[idx_boot]
            # get feature importance
            coef_boot[t,:] = func(x_boot, y_boot)
    return coef_boot

def _getCI(coef_boot, percent=.95, digit=4):
    # confidence interval over the columns of ndarray
    # coef_boot = (times, x.shape[1])
    ci = []
    for col in range(coef_boot.shape[1]):
        cicol = np.percentile(coef_boot[:,col], [50*(1-percent), 100-50*(1-percent)])
        cicol = [round(x, digit) for x in cicol]
        ci.append(cicol)
    return ci

def _getMean(coef_boot):
    # mean over columns, just to save a line
    return np.mean(coef_boot, axis=0)

def _returnTable(col_list):
    col_len = len(col_list)
    row_len = len(col_list[0])
    rows = []
    for i in range(row_len):
        rows.append('\t'.join([str(col[i]) for col in col_list]))
    return '\n'.join(rows)

def _getPercent(value, digit=2):
    # value could be list or ndarray (1D)
    # return list
    percent_list = value / np.sum(value)
    return [round(p, digit) for p in percent_list]


def printBootResult(coef_boot, names_full, names_selected):
    # Print for table filling
    # mean
    mean = _getMean(coef_boot)
    CI = [str(ci) for ci in _getCI(coef_boot)]
    rank = _getRank(mean, names_selected)['rank'].values.tolist()
    percent = _getPercent(mean)
    print(_returnTable([_fillBlank(col, names_full, names_selected) for col in [mean, CI, percent, rank]]))
    return

def printBootResultCA(result_df):
    print(_returnTable(result_df.transpose().values.tolist()))

def _fillBlank(content, names_full, names_selected):
    # Get full content of a list (a column) to fill the table
    num_full = len(names_full)
    num_selected = len(names_selected)
    full_content = ['NA'] * num_full
    # fill the rows with values
    for i in range(num_selected):
        row = names_full.index(names_selected[i])
        full_content[row] = content[i]
    return full_content

def _getPercent(score):
    return score / np.sum(score)

def _getPercentString(number):
    number = str(number)
    return "{0:,1%}".format(number)

def getFeatureNames(feature_descriptions):
    name_list = []
    for row in range(feature_descriptions.shape[0]):
        name_list.append('_'.join([str(x) for x in list(feature_descriptions.loc[row,:])]))
    return name_list


# debug function
def debug():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data[:,1:4])
    Y = iris.data[:, 0]

if __name__ == '__main__':
    debug()
