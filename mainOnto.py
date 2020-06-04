# pip install shap
# pip install --upgrade scikit-image
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import numpy as np
import Evaluator as E
import utils as U
import config as conf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import ShapHandler as S

feature_mapping = dict(zip(range(0, 99), ['Matrix_avg', 'Matrix_max', 'Matrix_std', 'Matrix_mpe', 'Matrix_mcd',
                                          'Matrix_bbm', 'Matrix_bpm', 'Matrix_lmm', 'Matrix_dom', 'Matrix_pca1',
                                          'Matrix_pca2', 'Matrix_pcaSum', 'Matrix_pcaEntropy', 'Matrix_norms1',
                                          'Matrix_norms2', 'Matrix_normsF', 'Matrix_normsInf', 'Mouse_totalLength',
                                          'Mouse_totalActions', 'Mouse_totalTime', 'Mouse_totalDist', 'Mouse_maxSpeed',
                                          'Mouse_minX', 'Mouse_minY', 'Mouse_maxX', 'Mouse_maxY', 'Mouse_avgSpeed',
                                          'Mouse_avgX', 'Mouse_avgY', 'Behavioural_countDistinctCorr',
                                          'Behavioural_countGeneralCorr', 'Behavioural_countMindChange',
                                          'Behavioural_avgConf', 'Behavioural_maxConf', 'Behavioural_minConf',
                                          'Behavioural_avgTime', 'Behavioural_maxTime', 'Behavioural_minTime',
                                          'Sequential_confP', 'Sequential_confR', 'Sequential_confRes',
                                          'Sequential_confCal',
                                          'Sequential_timeP', 'Sequential_timeR', 'Sequential_timeRes',
                                          'Sequential_timeCal',
                                          'Sequential_consensusP', 'Sequential_consensusR', 'Sequential_consensusRes',
                                          'Sequential_consensusCal', 'Spatial_MoveP', 'Spatial_MoveR',
                                          'Spatial_MoveRes',
                                          'Spatial_MoveCal', 'Spatial_LMouseP', 'Spatial_LMouseR', 'Spatial_LMouseRes',
                                          'Spatial_LMouseCal', 'Spatial_WMouseP', 'Spatial_WMouseR',
                                          'Spatial_WMouseRes',
                                          'Spatial_WMouseCal', 'Spatial_RMouseP', 'Spatial_RMouseR',
                                          'Spatial_RMouseRes',
                                          'Spatial_RMouseCal']))

feature_positions = {'Matrix': (1, 18),
                     'Mouse': (18, 30),
                     'Behavioural': (30, 39),
                     # 'Behavioural': (18, 39),
                     'Sequential': (39, 51),
                     'Spatial': (51, 67)}

folder_train = './results/18_07_2019_18_19 50-50/'
folder_test = './results/15_01_2020_11_56 onto/'
Y_train = pd.read_csv(folder_train + 'quality.csv')
Y_test = pd.read_csv(folder_test + 'quality.csv')
X_train = pd.read_csv(folder_train + 'features.csv')
X_test = pd.read_csv(folder_test + 'features.csv')

X_train = X_train.drop(list(map(str, list(range(66, len(X_train.columns)-1)))), axis=1)
X_test = X_test.drop(list(map(str, list(range(66, len(X_test.columns)-1)))), axis=1)
# X = X.drop(list(map(str, list(range(18, len(X.columns)-1)))), axis=1)
X_train.columns = ['matcher', ] + [feature_mapping[int(item)] for item in list(X_train.columns)[1:]]
X_test.columns = ['matcher', ] + [feature_mapping[int(item)] for item in list(X_test.columns)[1:]]

Y_train = pd.concat([Y_train, Y_test.iloc[:10]], ignore_index=True)
X_train = pd.concat([X_train, X_test.iloc[:10]], ignore_index=True)
Y_test = Y_test.iloc[10:]
X_test = X_test.iloc[10:]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]
# names = ["AdaBoost"]

clfs = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]
# clfs = [AdaBoostClassifier()]

classifiers = list(zip(names, clfs))
res = None
# kfold = KFold(conf.folds, True, 1)
# matchers = list(Y['matcher'].unique())
# matchers = list(Y[~Y['matcher'].str.contains('_')]['matcher'].unique())
# matchers_ids = dict(enumerate(matchers))

X_test = X_test.drop('matcher', axis=1)
X_train = X_train.drop('matcher', axis=1)
x_test = np.array(X_test)
x_train = np.array(X_train)

predictions = Y_test[['matcher', 'P_bin', 'R_bin', 'Res_bin', 'Cal_bin']]
yP_train = np.array(Y_train['P_bin'])
yR_train = np.array(Y_train['R_bin'])
yRes_train = np.array(Y_train['Res_bin'])
yCal_train = np.array(Y_train['Cal_bin'])
yP_test = np.array(Y_test['P_bin'])
yR_test = np.array(Y_test['R_bin'])
yRes_test = np.array(Y_test['Res_bin'])
yCal_test = np.array(Y_test['Cal_bin'])

for clf_name, clf in classifiers:
    print('Using ' + clf_name)
    if U.check_labels(yP_train):
        clf.fit(X=x_train, y=yP_train)
        predictions['P_bin_' + clf_name] = clf.predict(x_test)
    else:
        predictions['P_bin_' + clf_name] = len(x_test) * [yP_train[0], ]
    S.check_importance(X_train, yP_train, X_test, clf, 'P', clf_name, 'full', 1)

    if U.check_labels(yR_train):
        clf.fit(X=x_train, y=yR_train)
        predictions['R_bin_' + clf_name] = clf.predict(x_test)
    else:
        predictions['R_bin_' + clf_name] = len(x_test) * [yR_train[0], ]
    S.check_importance(X_train, yR_train, X_test, clf, 'R', clf_name, 'full', 1)

    if U.check_labels(yRes_train):
        clf.fit(X=x_train, y=yRes_train)
        predictions['Res_bin_' + clf_name] = clf.predict(x_test)
    else:
        predictions['Res_bin_' + clf_name] = len(x_test) * [yRes_train[0], ]
    S.check_importance(X_train, yRes_train, X_test, clf, 'Res', clf_name, 'full', 1)

    if U.check_labels(yCal_train):
        clf.fit(X=x_train, y=yCal_train)
        predictions['Cal_bin_' + clf_name] = clf.predict(x_test)
    else:
        predictions['Cal_bin_' + clf_name] = len(x_test) * [yRes_train[0], ]
    S.check_importance(X_train, yCal_train, X_test, clf, 'Cal', clf_name, 'full', 1)
res = pd.concat([res, predictions], ignore_index=True).drop_duplicates().reset_index(drop=True)

res.sort_values(by='matcher', ascending=True).to_csv(folder_test + '/classification_results.csv', index=False)
sums = E.summerize_results(res, classifiers)
sums.sort_values(by=['Q', 'Acc'], ascending=True).to_csv(folder_test + '/classification_evaluation.csv', index=False)

exit()
for q in ['P_bin', 'R_bin', 'Res_bin', 'Cal_bin']:
    print('Ablation test for ' + q)
    i = 1
    sumQ = sums[sums['Q'] == q]
    best_clf = sums.ix[sumQ['Acc'].astype(float).argmax()]['Clf']
    best_clf_acc = sumQ['Acc'].astype(float).max()
    clf = dict(classifiers)[best_clf]
    feat_ablation = pd.DataFrame(columns=['ExcludedFeatures', 'Acc'])
    feat_res = None
    feat_ablation.loc[i] = np.array([best_clf, best_clf_acc])
    i += 1
    for subset_prefix in ['Matrix', 'Mouse', 'Behavioural', 'Sequential', 'Spatial']:
        print('Removing ' + subset_prefix + ' features')
        preds = []
        reals = []
        features2drop = list(X.columns)[feature_positions[subset_prefix][0]: feature_positions[subset_prefix][1]]
        subX = X.copy()
        subX = subX.drop(features2drop, axis=1)
        for _, test in kfold.split(matchers):
            testset = [matchers_ids[m] for m in test]
            X_test = subX[subX['matcher'].isin(testset)].drop('matcher', axis=1)
            X_train = subX[~subX['matcher'].isin(testset)].drop('matcher', axis=1)
            x_test = np.array(X_test)
            x_train = np.array(X_train)
            y_train = np.array(Y[~Y['matcher'].isin(testset)][q])
            y_test = np.array(Y[Y['matcher'].isin(testset)][q])
            if U.check_labels(y_train):
                clf.fit(X=x_train, y=y_train)
                pred = clf.predict(x_test)
            else:
                pred = len(x_test) * [y_train[0], ]
            S.check_importance(X_train, y_train, X_test, clf, q, best_clf, 'without_' + subset_prefix, i)
            preds += list(pred)
            reals += list(y_test)
        predictions = Y[~Y['matcher'].str.contains('_')][['matcher', q]]
        predictions[best_clf + '_without_' + subset_prefix] = preds
        # feat_res = pd.concat([feat_res, predictions], ignore_index=True).drop_duplicates().reset_index(drop=True)
        feat_ablation.loc[i] = np.array([best_clf + '_without_' + subset_prefix, E.eval_model(preds, reals)])
        predictions.sort_values(by='matcher', ascending=True).to_csv(folder + q + '_without_' + subset_prefix + "_classification_results_abl.csv",
                                                                  index=False)
        i += 1
    for subset_prefix in ['Matrix', 'Mouse', 'Behavioural', 'Sequential', 'Spatial']:
        print('Maintaining only ' + subset_prefix + ' features')
        preds = []
        reals = []
        features2maintain = ['matcher', ] + list(X.columns)[feature_positions[subset_prefix][0]:
                                                            feature_positions[subset_prefix][1]]
        subX = X.loc[:, features2maintain].copy()
        for _, test in kfold.split(matchers):
            testset = [matchers_ids[m] for m in test]
            X_test = subX[subX['matcher'].isin(testset)].drop('matcher', axis=1)
            X_train = subX[~subX['matcher'].isin(testset)].drop('matcher', axis=1)
            x_test = np.array(X_test)
            x_train = np.array(X_train)
            y_train = np.array(Y[~Y['matcher'].isin(testset)][q])
            y_test = np.array(Y[Y['matcher'].isin(testset)][q])
            if U.check_labels(y_train):
                clf.fit(X=x_train, y=y_train)
                pred = clf.predict(x_test)
            else:
                pred = len(x_test) * [y_train[0], ]
            S.check_importance(X_train, y_train, X_test, clf, q, best_clf, 'with_' + subset_prefix, i)
            preds += list(pred)
            reals += list(y_test)
        feat_ablation.loc[i] = np.array([best_clf + '_only_' + subset_prefix, E.eval_model(preds, reals)])
        predictions = Y[~Y['matcher'].str.contains('_')][['matcher', q]]
        predictions[best_clf + '_with_' + subset_prefix] = preds
        # feat_res = pd.concat([feat_res, predictions], ignore_index=True).drop_duplicates().reset_index(drop=True)
        predictions.sort_values(by='matcher', ascending=True).to_csv(folder + q + '_with_' + subset_prefix + "_classification_results_abl.csv",
                                                                  index=False)
        i += 1
    feat_ablation.sort_values(by=['Acc'],
                              ascending=True).to_csv(folder + '/' + q + '_ablation_evaluation.csv', index=False)
