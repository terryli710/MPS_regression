## Calculate feature importance, but focus on "meta-features" which are cetegorized by
## rules from different perspectives: orders, directions, powers.

from util_relaimpo import *
from util_ca import *
from util import loadNpy, loadCsv

feature_description = loadCsv(['data', 'X', 'feature_descriptions.csv'])
feature_names = getFeatureNames(feature_description)

def dvdX(X, divided_by="", feature_description=feature_description):
    # x is a dataframe!
    assert isinstance(X, pd.DataFrame)
    x_list = []
    # by nothing
    if not divided_by: return [X]
    # by order
    elif divided_by.lower() == 'order': col_name = 'Order'
    # by direction
    elif divided_by.lower() == 'direction': col_name = 'Physics Variable'
    # by power
    elif divided_by.lower() == 'power': col_name = 'Power'
    feature_names = list(feature_description[col_name].unique())
    for item in feature_names:
        x_list.append(X.iloc[:,list(feature_description[col_name]==item)])
    return x_list, feature_names

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

def mainCA(x_name, y_name, divided_by = "", feature_names = []):
    X = loadNpy(['data', 'X', x_name])
    Y = loadNpy(['data', 'Y', y_name])
    # INFO
    print("Dataset", x_name.split('_')[0])
    print("Method: ", "CA")
    print("Divided by", divided_by)
    # make dataframe
    if feature_names: xdf = pd.DataFrame(data=X, columns=feature_names)
    else: xdf = pd.DataFrame(data=X)
    # divide X
    x_list, feature_names = dvdX(xdf, divided_by=divided_by)
    print("bootstrapping ...")
    coef_boot,comb_feature = bootstrappingCA(x_list, Y)
    result_df = caResultDf(comb_feature, coef_boot)
    printBootResultCA(result_df)

if __name__ == '__main__':
    # ca
    # mainCA('HM_X_ang_vel.npy','HM_CSDM.npy', "order", feature_names)
    # mainCA('HM_X_ang_vel.npy','HM_CSDM.npy', "direction", feature_names)
    # mainCA('AF_X_ang_vel.npy','AF_CSDM.npy', "order", feature_names)
    # mainCA('AF_X_ang_vel.npy','AF_CSDM.npy', "direction", feature_names)
    # mainCA('NFL53_X_ang_vel.npy','NFL53_CSDM.npy', "order", feature_names)
    # mainCA('NFL53_X_ang_vel.npy','NFL53_CSDM.npy', "direction", feature_names)

    # other methods
    method = first
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', method, "order", feature_names)
    main('AF_X_ang_vel.npy', 'AF_MPS95.npy', method, "order", feature_names)
    main('NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', method, "order", feature_names)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', method, "direction", feature_names)
    main('AF_X_ang_vel.npy', 'AF_MPS95.npy', method, "direction", feature_names)
    main('NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', method, "direction", feature_names)
    main('HM_X_ang_vel.npy', 'HM_MPS95.npy', method, "power", feature_names)
    main('AF_X_ang_vel.npy', 'AF_MPS95.npy', method, "power", feature_names)
    main('NFL53_X_ang_vel.npy', 'NFL53_MPS95.npy', method, "power", feature_names)