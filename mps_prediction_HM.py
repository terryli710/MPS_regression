# this file builds regression models to predict 95% mps

from util import *
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from model_constants import *

def main(x_data, y_data, feature_eng, param_list):
    # load data
    x_raw = loadNpy(['data','X',x_data])
    y = loadNpy(['data','Y', y_data])

    # split data
    x_train_raw, x_test_raw = splitTrainTest(x_raw, verbose=False)
    y_train, y_test = splitTrainTest(y, verbose=False)

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
    # saveLog([x_data, y_data], ss.mean_, ss.var_, feature_eng, best_param, cv_results, preds, evaluate, boot_results, verbose=False)

    # Print results
    fillTable(cv_results, evaluate, boot_results, digit = 6)

if __name__ == '__main__':
    # data
    data_dict = {'x_data': 'HM_X_ang_vel.npy', 'y_data': 'HM_MPS95.npy'}

    # main loop
    for param in main_param_list:
        pdict = mergeDict(data_dict, param)
        main(**pdict)
