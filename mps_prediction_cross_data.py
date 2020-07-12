# this file builds regression models to predict 95% mps and other Ys using two data sets
from util import *
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from model_constants import *

def main(x_train_data, y_train_data, x_test_data, y_test_data, feature_eng, param_list):
    # load data
    # training set
    x_train_raw = loadNpy(['data','X',x_train_data])
    y_train = loadNpy(['data','Y', y_train_data])
    # testing set
    x_test_raw = loadNpy(['data','X',x_test_data])
    y_test = loadNpy(['data', 'Y', y_test_data])

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
    # saveLog([x_train_data, y_train_data, x_test_data, y_test_data], ss.mean_, ss.var_, feature_eng, best_param, cv_results, preds, evaluate, boot_results, verbose=False)

    # Print results
    fillTable(cv_results, evaluate, boot_results, digit = 6)

if __name__ == '__main__':
    #data
    data_dict = {'x_train_data':'HM_X_ang_vel.npy', 'y_train_data':'HM_MPS95.npy',
     'x_test_data':'AF_X_ang_vel.npy', 'y_test_data':'AF_MPS95.npy'}
    # main loop
    for param in main_param_list:
        pdict = mergeDict(data_dict, param)
        main(**pdict)


    '''main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', '', simple_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', '', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', '', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.95', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.95', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.90', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.90', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.80', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'PCA-0.80', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-f_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-f_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-f_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-f_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-mutual_info_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-mutual_info_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-mutual_info_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'GUS-mutual_info_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'FPR-f_regression-0.05', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'AF_X_ang_vel.npy', 'AF_MPS95.npy', 'FPR-f_regression-0.05', lasso_param_list)'''


    '''main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', '', simple_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', '', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', '', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.95', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.95', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.90', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.90', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.80', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'PCA-0.80', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-f_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-f_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-f_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-f_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-mutual_info_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-mutual_info_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-mutual_info_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'GUS-mutual_info_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'FPR-f_regression-0.05', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', 'FPR-f_regression-0.05', lasso_param_list)'''

    '''main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', '', simple_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', '', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', '', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.95', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.95', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.90', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.90', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.80', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'PCA-0.80', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-f_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-f_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-f_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-f_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-mutual_info_regression-30', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-mutual_info_regression-30', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-mutual_info_regression-10', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'GUS-mutual_info_regression-10', lasso_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'FPR-f_regression-0.05', ridge_param_list)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', 'PAC12_X_ang_vel.npy', 'PAC12_MPS95.npy', 'FPR-f_regression-0.05', lasso_param_list)'''