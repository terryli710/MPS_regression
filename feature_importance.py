## Feature Importance of Brain Strain Linear Model\

# The workflow looks like the following.
#
# 1. Build a linear model over the dataset by:
#   - Feature selection using VIF until there is no severe collinearity effect
#   - Use no other feature selection method, taking all the variables into the model
#   - First use least squared fit, then consider huber fit
# 2. Find feature importance of the model by:
#   - Using `first`, `last` and `standard coefficients` methods
#   - Test the robustness of this ranking by:
#     - Compare results in different dataset;
#     - Acquired by different methods

from util_relaimpo import *
from util import loadNpy

def main(x_name, y_name, method, feature_names = []):
    X = loadNpy(['data','X',x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # standardize before vif
    ss = StandardScaler()
    x_std = ss.fit_transform(X)
    # make dataframe
    if feature_names: xdf = pd.DataFrame(data=x_std, columns=feature_names)
    else: xdf = pd.DataFrame(data = x_std)
    x_selected_std, vif = vifStepwiseSelect(xdf)
    coef_boot =  bootstrapping(x_selected_std, Y, method)
    print(printBootResult(coef_boot, list(xdf.columns), list(x_selected_std.columns)))

if __name__ == '__main__':
    main('HM_X_ang_vel.npy','HM_MPS95.npy',structcoef)

