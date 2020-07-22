# This file contains util functions to calculate relative importance in linear models

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


# debug function
def debug():
    from util import loadNpy
    X = loadNpy(['data','X','HM_X_ang_vel.npy'])
    xdf = pd.DataFrame(X)
    x_selected, vif = vifStepwiseSelect(xdf)
    print(x_selected)
