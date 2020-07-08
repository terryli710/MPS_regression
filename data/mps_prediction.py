# this file builds regression models to predict 95% mps
from util import loadNpy, splitTrainTest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # load data
    HM_ang_acc = loadNpy(['data','X','HM_ang_acc_40.npy'])
    HM_ang_vel = loadNpy(['data', 'X', 'HM_ang_vel_40.npy'])
    HM_lin_acc = loadNpy(['data', 'X', 'HM_lin_acc_40.npy'])
    x_raw = np.concatenate((HM_lin_acc, HM_ang_vel, HM_ang_acc), axis=1)
    y = loadNpy(['data','Y', 'HM_95.npy'])

    # split data
    x_train_raw, x_test_raw = splitTrainTest(x_raw, verbose=True)
    y_train, y_test = splitTrainTest(y, verbose=True)

    # standardize the data
    ss = StandardScaler()
    ss.fit(x_train_raw)
    x_train_s = ss.transform(x_train_raw, copy=True)
    x_test_s = ss.transform(x_test_raw, copy=True)
    print("mean is {}, var is {}".format(ss.mean_, ss.var_))
    # possible hyper-parameter tuning using cross-validation

    # tuned model trained on training set.
    lr = LinearRegression()
    lr.fit(x_train_s, y_train)
    # evaluated on test set.
    preds = lr.predict(x_test_s)
    mean_absolute_error(y_test, preds)
    mean_squared_error(y_test, preds)
    r2_score(y_test, preds)


if __name__ == '__main__':
    main()