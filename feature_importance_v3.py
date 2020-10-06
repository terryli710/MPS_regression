## Calculate feature importance, but focus on "meta-features" which are categorized by
## rules from different perspectives: orders, directions, powers.

## for "simple methods"

from util_relaimpo import *
from util_ca import *
from util import loadNpy

def main(x_name, y_name, method, divided_by = "", feature_names = []):
    # INFO
    print("Dataset", x_name.split('_')[0])
    print("Method", str(method).split(' ')[1])
    # load data
    X = loadNpy(['data', 'X', x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # make dataframe
    if feature_names: xdf = pd.DataFrame(data=X, columns=feature_names)
    else: xdf = pd.DataFrame(data=X)
    # divide X
    x_list, feature_names = dvdX(xdf, divided_by=divided_by)
    print("bootstrapping ...")
    coef_boot = bootstrapping(x_list, Y, method)
    printBootResult(coef_boot, list(feature_names), list(feature_names))
    pt = tTestTopTwo(coef_boot)
    pa = anovaBoot(coef_boot)
    print(returnTable([['t-test'], [str(pt)], ['ANOVA'], [str(pa)]]))


if __name__ == '__main__':
    # first and structcoef
    x_prefix = ["HM", "AF", "NFL", "PAC",  "MMA", "NHTSA", "NASCAR"]
    y_suffix = ["MPS95", "MPSCC95", "CSDM"]
    x_main = "{}_X_ang_vel.npy"
    y_main = "{}_{}.npy"
    divided_list = ["order", "direction", "power"]
    methods = [first, structcoef]
    for ys in y_suffix:
        for xp in x_prefix:
            for method in methods:
                for divide in divided_list:
                    x_name = x_main.format(xp)
                    y_name = y_main.format(xp, ys)
                    main(x_name, y_name, method, divide, feature_names)
