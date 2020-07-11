# this file builds regression models to predict 95% mps



from util import *
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

def main(x_data, y_data, feature_eng, param_list):
    # load data
    x_raw = loadNpy(['data','X',x_data])
    y = loadNpy(['data','Y', y_data])

    # split data
    x_train_raw, x_test_raw = splitTrainTest(x_raw, verbose=True)
    y_train, y_test = splitTrainTest(y, verbose=True)

    # standardize the data
    ss = StandardScaler()
    ss.fit(x_train_raw)
    x_train_s = ss.transform(x_train_raw, copy=True)
    x_test_s = ss.transform(x_test_raw, copy=True)
    # print("mean is {}, var is {}".format(ss.mean_, ss.var_))

    # hyper-parameter tuning using cross-validation
    best_param = outputBestParam(ElasticNet, param_list, x_train_s, y_train, feature_eng)
    lr = ElasticNet(**best_param)

    # CV
    cv_results = crossValidation(lr, x_train_s, y_train, feature_eng, fold=5)

    # tuned model trained on training set.
    preds, evaluate = modelTrainTest(lr, x_train_s, y_train, x_test_s, y_test, feature_eng)

    # bootstrap
    boot_results = bootstrap(lr, x_train_s, y_train, x_test_s, y_test, feature_eng, times=30)

    # Store results
    saveLog(x_data, y_data, ss.mean_, ss.var_, feature_eng, best_param, cv_results, preds, evaluate, boot_results,verbose=True)

    # Print results
    fillTable(cv_results, evaluate, boot_results)

if __name__ == '__main__':
    lasso_param_list = [{'alpha': 1e-3, 'l1_ratio': 1.0},
                        {'alpha': 1e-4, 'l1_ratio': 1.0},
                        {'alpha': 1e-5, 'l1_ratio': 1.0},]
    ridge_param_list = [{'alpha': 1e-3, 'l1_ratio': 0.0},
                        {'alpha': 1e-4, 'l1_ratio': 0.0},
                        {'alpha': 1e-5, 'l1_ratio': 0.0}]
    simple_param_list = [{'alpha': 1e-7, 'l1_ratio': 0.0}]
    param_list = [{'alpha': 1e-3, 'l1_ratio': 1.0},
                  {'alpha': 1e-4, 'l1_ratio': 1.0},
                  {'alpha': 1e-5, 'l1_ratio': 1.0},
                  {'alpha': 1e-3, 'l1_ratio': 0.0},
                  {'alpha': 1e-4, 'l1_ratio': 0.0},
                  {'alpha': 1e-5, 'l1_ratio': 0.0}]
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', '', simple_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', '', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', '', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.95', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.95', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.90', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.90', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.80', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PCA-0.80', lasso_param_list)

