# This file stores constants for training experiments
from sklearn.linear_model import HuberRegressor

lasso_param_list = [{'alpha': 1e-1, 'l1_ratio': 1.0},
                    {'alpha': 1e-2, 'l1_ratio': 1.0},
                    {'alpha': 1e-3, 'l1_ratio': 1.0},
                    {'alpha': 1e-4, 'l1_ratio': 1.0},
                    {'alpha': 1e-5, 'l1_ratio': 1.0}]
ridge_param_list = [{'alpha': 1e-1, 'l1_ratio': 0.0},
                    {'alpha': 1e-2, 'l1_ratio': 0.0},
                    {'alpha': 1e-3, 'l1_ratio': 0.0},
                    {'alpha': 1e-4, 'l1_ratio': 0.0},
                    {'alpha': 1e-5, 'l1_ratio': 0.0}]
simple_param_list = [{'alpha': 1e-7, 'l1_ratio': 0.0}]
huber_param_list = [{'alpha': 1e-1},
                    {'alpha': 1e-2},
                    {'alpha': 1e-3},
                    {'alpha': 1e-4},
                    {'alpha': 1e-5}]
param_list = [{'alpha': 1e-3, 'l1_ratio': 1.0},
              {'alpha': 1e-4, 'l1_ratio': 1.0},
              {'alpha': 1e-5, 'l1_ratio': 1.0},
              {'alpha': 1e-3, 'l1_ratio': 0.0},
              {'alpha': 1e-4, 'l1_ratio': 0.0},
              {'alpha': 1e-5, 'l1_ratio': 0.0}]


# experiments to complete for a comb of data
main_param_list = [{'feature_eng': '', 'param_list': simple_param_list},
                   {'feature_eng': '', 'param_list': ridge_param_list},
                   {'feature_eng': '', 'param_list': lasso_param_list},
                   {'feature_eng': 'PCA-0.95', 'param_list': ridge_param_list},
                   {'feature_eng': 'PCA-0.95', 'param_list': lasso_param_list},
                   {'feature_eng': 'PCA-0.90', 'param_list': ridge_param_list},
                   {'feature_eng': 'PCA-0.90', 'param_list': lasso_param_list},
                   {'feature_eng': 'PCA-0.80', 'param_list': ridge_param_list},
                   {'feature_eng': 'PCA-0.80', 'param_list': lasso_param_list},
                   {'feature_eng': 'GUS-f_regression-30', 'param_list': ridge_param_list},
                   {'feature_eng': 'GUS-f_regression-30', 'param_list': lasso_param_list},
                   {'feature_eng': 'GUS-f_regression-10', 'param_list': ridge_param_list},
                   {'feature_eng': 'GUS-f_regression-10', 'param_list': lasso_param_list},
                   {'feature_eng': 'GUS-mutual_info_regression-30', 'param_list': ridge_param_list},
                   {'feature_eng': 'GUS-mutual_info_regression-30', 'param_list': lasso_param_list},
                   {'feature_eng': 'GUS-mutual_info_regression-10', 'param_list': ridge_param_list},
                   {'feature_eng': 'GUS-mutual_info_regression-10', 'param_list': lasso_param_list},
                   {'feature_eng': 'FPR-f_regression-0.05', 'param_list': ridge_param_list},
                   {'feature_eng': 'FPR-f_regression-0.05', 'param_list': lasso_param_list},
                   {'feature_eng': '', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'PCA-0.95', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'PCA-0.90', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'PCA-0.80', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'GUS-f_regression-30', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'GUS-f_regression-10', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'GUS-mutual_info_regression-30', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'GUS-mutual_info_regression-10', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   {'feature_eng': 'FPR-f_regression-0.05', 'param_list': huber_param_list, 'lr_model' : HuberRegressor},
                   ]