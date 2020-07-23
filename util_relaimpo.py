# This file contains util functions to calculate relative importance in linear models

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations, compress

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
            variable_delete = int(max_vif['variable'])
            x_copy = x_copy.drop(variable_delete, axis=1)
            deleted.append(variable_delete)
            i += 1
        else:
            if verbose:
                print('Deleted variable index: ', deleted)
            return x_copy, vif
        if verbose:
            print("Iteration ", i)
            if verbose > 1:
                print("Problematic variables and vifs are: ")
                print(oob_vif)
                print("Problematic variables are: ", list(oob_vif.variable))
            else:
                print("Problematic variables are: ", list(oob_vif.variable))
            print("Delete variable NO.",int(max_vif['variable']))

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


def first(x, y):
    # return the R^2 of p linear models
    # model i: y = beta * x_i + intercept
    r2_list = []
    for i in range(x.shape[1]):
        lr = LinearRegression()
        lr.fit(x[:,i:i+1], y)
        yhat = lr.predict(x[:,i:i+1])
        r2_list.append(_r2(y, yhat))
    return r2_list

# 2. srw
def _stdLinear(x, y):
    ss = StandardScaler()
    xs = ss.fit_transform(x)
    lr = LinearRegression()
    lr.fit(xs, y)
    return lr, ss

def srw(x, y):
    lr,_ = _stdLinear(x, y)
    return list(lr.coef_[0,:])

# 3. structcoef
def structcoef(x, y):
    lr = LinearRegression()
    lr.fit(x, y)
    yhat = lr.predict(x)
    r2_total = _r2(y, yhat)
    return first(x, y) / r2_total

# 4. pratt
def pratt(x, y):
    beta = srw(x, y)
    r2 = first(x, y)
    return [a*b for a,b in zip(beta, r2)]

# 5. Commonality Analysis (ca)
def _subsetMatrix(num):
    # compute subset matrix:
    # number of features = k = num
    # matrix dimension: (k, 2^k-1)
    matrix = np.zeros((num, 2**num-1), dtype=bool)
    index=0
    for num_variable in range(1, num+1):
        for comb in combinations(range(num), num_variable):
            for value in comb:
                matrix[value,index]=True
            index+=1
    return matrix

def _getR2(X, Y):
    # get R^2 from a simple linear regression between X and Y
    lr = LinearRegression()
    lr.fit(X, Y)
    return _r2(Y, lr.predict(X))

def _list2Name(list):
    # convert list of columns to a name
    return ','.join(map(str, list))

def _aps(X,Y):
    '''
    Perform all possible subset regression to all the variables in X
    :return: a DF of [index, feature name, R2]
    '''
    # get feature name
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    else:
        feature_names = list(range(X.shape[1]))
    # get subset matrix
    feature_num = X.shape[1]
    subset_matrix = _subsetMatrix(feature_num)
    # regression models and compute R^2
    index = list(range(2**X.shape[1]-1))
    comb_feature = []
    r2_score = []
    for i in index:
        comb_feature.append(list(compress(feature_names, subset_matrix[:,i])))
        x_subset = X[:,subset_matrix[:,i]]
        r2_score.append(_getR2(x_subset,Y))
    comb_feature_names = [_list2Name(name) for name in comb_feature]
    r2_df = pd.DataFrame({'index':index, 'feature names':comb_feature_names, 'r2':r2_score})
    subset_df = pd.DataFrame(data=subset_matrix, index=feature_names, columns=comb_feature_names)
    return r2_df, subset_df, comb_feature

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
def _commonality(r2_df, subset_df, comb_feature):
    # basic ingredients
    nvar = len(subset_df.index)
    ivID = _ivID(subset_df)['ivID'].to_list()
    aps_bit_map =  _apsBitMap(subset_df, comb_feature)
    commonmality_list = []
    numcc = 2**nvar - 1

    ## Use the bitmap matrix to generate the list of R2 values needed.
    for i in range(numcc):
        bit = subset_df.iloc[0,i]
        if bit==1 : ilist = [0,-int(ivID[0])]
        else: ilist = [int(ivID[0])]
        for j in range(1,nvar):
            bit = subset_df.iloc[j,i]
            if bit==1 :
                alist = ilist
                blist = _genList(ilist, -ivID[j])
                ilist = alist + blist
            else: ilist = _genList(ilist,int(ivID[j]))
        ilist = [-x for x in ilist]
        commonmality_list.append(ilist)
    print(commonmality_list)


    ## Use the list of R2 values to compute each commonality coefficient.
    r2_matrix = r2_df['r2'].to_numpy()
    result_coef, result_percent = [], []
    for i in range(numcc):
        r2list = commonmality_list[i]
        numlist = len(r2list)
        ccsum=0
        for j in range(numlist):
            indexs = r2list[j]
            indexu = abs(indexs)
            if indexu != 0 :
                ccvalue = r2_matrix[indexu-1]
                if indexs < 0 : ccvalue *= -1
                ccsum += ccvalue
        result_coef.append(ccsum)
    return result_coef

def _caResultDf(comb_feature, result_coef, digit=5):
    percent = result_coef/np.sum(result_coef)
    return pd.DataFrame({'feature names': comb_feature,
                         'coefficients': [round(x, digit) for x in result_coef],
                         'percent': [round(x, digit) for x in percent]})

def ca(X,Y):
    '''
    Perform Commonality Analysis
    :return: [feature names, coefficients, % Total]
    '''
    r2_df, subset_df, comb_feature = _aps(X, Y)
    result_coef = _commonality(r2_df, subset_df, comb_feature)
    return _caResultDf(comb_feature, result_coef)

# debug function
def debug():
    from util import loadNpy
    X = loadNpy(['data','X','HM_X_ang_vel.npy'])
    Y = loadNpy(['data', 'Y', 'HM_MPS95.npy'])
    a = X[0:5,0:3]
    y = Y[0:5,:]
    b = ca(a, y)
    print(b)


if __name__ == '__main__':
    debug()