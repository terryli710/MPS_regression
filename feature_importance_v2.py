## Without filtering results with VIF, calculate the importance for all the features.

from util_relaimpo import *
from util import loadNpy, loadCsv

def main(x_name, y_name, method, feature_names = []):
    X = loadNpy(['data', 'X', x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # make dataframe
    if feature_names: xdf = pd.DataFrame(data=X, columns=feature_names)
    else: xdf = pd.DataFrame(data=X)
    print("bootstrapping ...")
    coef_boot = bootstrapping(xdf, Y, method)
    print(printBootResult(coef_boot, list(xdf.columns), list(xdf.columns)))

feature_names = getFeatureNames(loadCsv(['data', 'X', 'feature_descriptions.csv']))

if __name__ == '__main__':
    main('HM_X_ang_vel.npy','HM_MPS95.npy', structcoef, feature_names)
    main('AF_X_ang_vel.npy', 'AF_MPS95.npy', structcoef, feature_names)
    main('NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', structcoef, feature_names)
