# commonality analysis
import pandas as pd
import numpy as np
from itertools import combinations, compress
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from util_relaimpo import _r2, _getPercent, _getRank, _getMean, _getCI, _getR2

def _subsetMatrix(num, df=False, cf=False):
    # compute subset matrix:
    # number of features = k = num
    # matrix dimension: (k, 2^k-1)
    matrix = np.zeros((num, 2**num-1), dtype=bool)
    index=0
    feature_names = list(range(num))
    comb_features = []
    for num_variable in range(1, num+1):
        for comb in combinations(range(num), num_variable):
            comb_features.append(list(comb))
            for value in comb:
                matrix[value,index]=True
            index+=1
    if df:
        comb_feature_names = [_list2Name(name) for name in comb_features]
        subset_df = pd.DataFrame(data=matrix, index=feature_names, columns=comb_feature_names)
        return matrix, subset_df
    elif cf:
        return matrix, comb_features
    return matrix

def _list2Name(list):
    # convert list of columns to a name
    return ','.join(map(str, list))

def aps(X,Y):
    '''
    Perform all possible subset regression to all the variables in X
    X could be a list of df which denotes meta-features
    :return: a DF of [index, feature name, R2]
    '''
    # version 1, X = DF
    if isinstance(X, pd.DataFrame): feature_num = X.shape[1]
    # version 2, X = list
    elif isinstance(X, list): feature_num = len(X)
    else: raise TypeError('Wrong X type')
    # get subset matrix and df
    subset_matrix, subset_df = _subsetMatrix(feature_num, True)
    r2_list = _regressionSubset(X, Y, subset_matrix)
    r2_df = pd.DataFrame({'index': list(range(2**feature_num-1)),
                          'feature names': list(subset_df.columns),
                          'r2': r2_list})
    return r2_df, subset_df

def _aps(X, Y):
    '''
        Perform all possible subset regression to all the variables in X
        X could be a list of df which denotes meta-features
        :return: a DF of [index, feature name, R2]
        '''
    # version 1, X = DF
    if isinstance(X, pd.DataFrame):
        feature_num = X.shape[1]
    # version 2, X = list
    elif isinstance(X, list):
        feature_num = len(X)
    else:
        raise TypeError('Wrong X type')
    # get subset matrix and df
    subset_matrix, comb_feature = _subsetMatrix(feature_num, cf=True)
    r2_list = _regressionSubset(X, Y, subset_matrix)
    return r2_list, comb_feature

def _regressionSubset(X, Y, subset_matrix):
    # perform regression using a subset of X
    # return r2 of these regressions
    # version 1, X = DF
    if isinstance(X, pd.DataFrame):
        feature_num = X.shape[1]
        # regression models and compute R^2
        index = list(range(2**feature_num-1))
        r2_score = []
        for i in index:
            x_subset = X.iloc[:,subset_matrix[:,i]]
            r2_score.append(_getR2(x_subset,Y))
        return r2_score

    # version 2, X = list
    elif isinstance(X, list):
        feature_num = len(X)
        # regression models and compute R^2
        index = list(range(2**feature_num-1))
        r2_score = []
        for i in index:
            x_subset = pd.concat(list(compress(X, subset_matrix[:,i])),axis=1)
            r2_score.append(_getR2(x_subset, Y))
        return r2_score

def _ivID(subset_df):
    # get iv ID in a wierd way, refer to https://rdrr.io/cran/yhat/src/R/aps.r
    feature_list = subset_df.index
    ivID = [2**x for x in range(len(feature_list))]
    return pd.DataFrame(data=ivID, index=feature_list, columns=['ivID'])

def _apsBitMap(subset_df, comb_feature):
    ivID = _ivID(subset_df)
    bit = []
    for feature_list in comb_feature:
        value = 0
        for feature in feature_list:
            value += ivID.loc[feature, 'ivID']
        bit.append(value)
    return pd.DataFrame({'feature names':comb_feature, 'bit':bit})

def _genList(ivlist, value):
    nvar = len(ivlist)
    newlist = []
    for i in range(nvar):
        newlist.append(abs(ivlist[i])+abs(value))
        if (((ivlist[i]<0) and (value >= 0)) or ((ivlist[i]>=0) and (value <0))): newlist[i]*=-1
    return newlist

#EffectBitMap = subset_df
def _commonality(r2_list, int_matrix):
    # input r2_score list
    # output list of same length, but calculated based on commonality transfer matrix
    # basic ingredients
    return list(np.dot(int_matrix, r2_list))

def caResultDf(result_coef, comb_feature, digit=4):
    # version 1: a single time
    if isinstance(result_coef, list):
        percent = _getPercent(result_coef)
        return pd.DataFrame({'feature names': comb_feature,
                             'coefficients': [round(x, digit) for x in result_coef],
                             'percent': [round(x, digit) for x in percent]})
    # version 2: boot result
    elif isinstance(result_coef, np.ndarray):
        mean = _getMean(result_coef)
        CI = [str(ci) for ci in _getCI(result_coef)]
        rank = _getRank(mean, False, comb_feature)['rank'].values.tolist()
        percent = _getPercent(mean)
        return pd.DataFrame({'feature names': comb_feature,
                             'mean': [round(x, digit) for x in mean],
                             'CI': CI,
                             'percentage':percent,
                             'rank':rank})

def _ca(X, Y):
    #  Perform Commonality Analysis
    #  Perform Commonality Analysis
    r2_df, comb_features = _aps(X, Y)
    num = len(comb_features[-1])
    int_matrix = _getComMatrix(num)
    result_coef = _commonality(r2_df, int_matrix)
    return result_coef, comb_features

def ca(X,Y,names=[]):
    '''
    Perform Commonality Analysis
    :return: [feature names, coefficients, % Total]
    '''
    result_coef, comb_features = _ca(X, Y)
    return caResultDf(result_coef, comb_features, names=[])

def bootstrappingCA(x, y, times=100):
    # version 1
    if isinstance(x, pd.DataFrame):
        num_feature = 2**x.shape[1]-1
        coef_boot = np.zeros((times, num_feature))
        num_sample = x.shape[0]
        for t in tqdm(range(times)):
            # boot sample
            idx_boot = np.random.randint(0, num_sample, size=num_sample)
            x_boot = x.iloc[idx_boot, :]
            y_boot = y[idx_boot]
            # get feature importance
            coef_boot[t,:],comb_feature = _ca(x_boot, y_boot)
        return coef_boot, comb_feature
    # version 2
    elif isinstance(x, list):
        num_feature = 2**len(x)-1
        coef_boot = np.zeros((times, num_feature))
        num_sample = x[0].shape[0]
        for t in tqdm(range(times)):
            # boot sample
            idx_boot = np.random.randint(0, num_sample, size=num_sample)
            x_boot = [x_item.iloc[idx_boot, :] for x_item in x]
            y_boot = y[idx_boot]
            # get feature importance
            coef_boot[t,:], comb_feature = _ca(x_boot, y_boot)
        return coef_boot, comb_feature

def _getComMatrix(num):
    '''
    return a matrix (ndarray) that map Individual Sets and Unions to Intersections and complements
    :param num: num of elements in this case
    '''
    # basic ingredients
    subset_matrix, comb_features = _subsetMatrix(num, cf=True)
    ncomb = 2**num - 1

    # commonality matrix
    com_matrix = np.zeros((ncomb, ncomb))
    # intersection matrix
    int_matrix = _getIntersectionMatrix(num)

    # first order rows are the same
    com_matrix[:num, :] = int_matrix[:num, :]

    # higher orders
    for index in range(num,ncomb):
        _getExclusive(index, com_matrix, int_matrix, comb_features)

    return com_matrix

def _getExclusive(index, com_matrix, int_matrix, comb_features):
    # add a line of commonality matrix, (exclusive sets) according to intersection matrix
    elements = comb_features[index]
    order_ele = len(elements)
    if order_ele == 1: raise ValueError("Order should be greater than 1")
    order_total = len(comb_features[-1])
    # for all the orders
    for o in range(order_ele, order_total+1):
        for comb in combinations(comb_features[-1], o):
            # if this combination contains the elements we care
            if not set(elements) - set(comb):
                index_ele = comb_features.index(list(comb))
                com_matrix[index,:] += (-1)**(o-order_ele) * int_matrix[index_ele,:]

def _getIntersectionMatrix(num):
    # get intersection matrix
    # columns are unions, rows are intersections
    # e.g. union [0], [1], [0,1]
    # int[0]     0    -1     1
    # int[1]    -1     0     1
    # int[0,1]   1     1    -1
    # basic ingredients
    subset_matrix, comb_features = _subsetMatrix(num, cf=True)
    ncomb = 2 ** num - 1
    # intersection matrix
    int_matrix = np.zeros((ncomb, ncomb))
    # fill up the matrix
    for i in range(ncomb):
        _getIntersection(i, int_matrix, comb_features)
    return int_matrix

def _getIntersection(index, int_matrix, comb_features):
    # add a line of intersection matrix according to previous information
    elements = comb_features[index]
    order = len(elements)
    if order == 1:
        int_matrix[index, -1] = 1
        comp_eles = _complementList(comb_features[-1], elements)
        comp_index = comb_features.index(comp_eles)
        int_matrix[index, comp_index] = -1
    elif order > 1:
        # order > 1
        # first add up first order
        for ele in elements:
            ele_index = comb_features.index([ele])
            int_matrix[index, ele_index] = 1
        # middle orders, from intersection
        for o in range(2, order):
            for comb in combinations(elements, o):
                ele_index = comb_features.index(list(comb))
                int_matrix[index, :] += -((-1)**o) * int_matrix[ele_index, :]
        # determine sign
        int_matrix[index, :] *= (-1)**order
        # final order
        ele_index = comb_features.index(list(elements))
        int_matrix[index, ele_index] = -((-1) ** order)

def _complementList(la, lb):
    # find la that is not in lb
    lc = []
    for ele in la:
        if ele not in lb:
            lc.append(ele)
    return lc


# debug function
def debug():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data[:, 1:4])
    Y = iris.data[:, 0]
    a = _ca(X, Y)
    caResultDf(a[0], a[1])

if __name__ == '__main__':
    debug()