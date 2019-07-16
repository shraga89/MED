import mouseHandler as MH
import utils as U
import HHandler as HH
import Evaluator as E
import config as conf
import MatchingFeatures as MF
# import glob
from os import listdir
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import random, time, datetime
import keras
from keras import backend as K
import tensorflow as tf
import pandas as pd
import os
from keras.applications.resnet50 import preprocess_input

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 28})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
keras.backend.set_session(sess)

matchers = listdir(str(conf.dir + 'ExperimentData/'))
print('found ', len(matchers), ' matchers')
matchers_ids = dict(enumerate(matchers))
# for matcher in matchers:
#     test = MH.mouseHandler(matcher)
#     test.exportMouseData('LMouse')
#     test.exportMouseData('RMouse')
#     test.exportMouseData('WMouse')
#     test.exportMouseData('Move')

# SHAP to analyze features

evaluator = E.Evaluator()
quality = {}
features = {}
match_seqs = {}
conf_seqs = {}
time_seqs = {}
consensus_seqs = {}
submatchers = {}
matches = {}
heatmaps = {'Move': {}, 'LMouse': {}, 'WMouse': {}, 'RMouse': {}}
matcher_count = 1
matcher_number = len(matchers)+1
res = None
kfold = KFold(conf.folds, True, 1)
for matcher in matchers:
    if matcher_count >= matcher_number:
        break
    matcher_count += 1
    print('Matcher Number', matcher)
    Hmatcher = HH.HHandler(matcher)
    match = Hmatcher.getMatch()
    match_seqs[matcher], conf_seqs[matcher], time_seqs[matcher] = Hmatcher.getSeqs()
    features[matcher] = MF.extractPreds(evaluator.getMatrix4Match(match))
    quality[matcher] = evaluator.evaluate(match)
    mouse = MH.mouseHandler(matcher)
    for k in heatmaps:
        heatmaps[k][matcher] = mouse.exportMouseData(k)
    curr_feat = features[matcher].copy()
    temp = mouse.extract_mouse_features()
    features[matcher] = np.append(curr_feat, temp)
    curr_feat = features[matcher].copy()

    temp = Hmatcher.extract_behavioural_features()
    features[matcher] = np.append(curr_feat, temp)
    matches[matcher] = match
    for i in range(conf.num_subs, min(len(conf_seqs[matcher]), conf.max_subs), conf.jumps):
        submatchers = Hmatcher.split2ns(i, matcher)
        submouses = mouse.split2ns(submatchers)
        for sub in submatchers:
            match = submatchers[sub].getMatch()
            match_seqs[sub], conf_seqs[sub], time_seqs[sub] = submatchers[sub].getSeqs()
            features[sub] = MF.extractPreds(evaluator.getMatrix4Match(match))
            for k in heatmaps:
                heatmaps[k][sub] = submouses[sub].exportMouseData(k)
            curr_feat = features[sub].copy()
            temp = submouses[sub].extract_mouse_features()
            features[sub] = np.append(curr_feat, temp)
            curr_feat = features[matcher].copy()
            temp = submatchers[sub].extract_behavioural_features()
            features[sub] = np.append(curr_feat, temp)
            quality[sub] = evaluator.evaluate(match)
Y = E.quality2pandas(quality, True)
i = 1
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
print(st)
matches_train = {}
for _, testset in kfold.split(matchers):
    test = [matchers_ids[m] for m in testset]
    matches_train = {k: matches[k] for k in matches if k not in test}
    st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
    print("Starting fold " + str(i) + ' ' + str(st))
    K.get_session().close()
    K.set_session(tf.Session())
    K.get_session().run(tf.global_variables_initializer())
    K.clear_session()
    U.load_vgg()
    consensus = U.bulid_consensus(matches_train)
    lstm_p = U.build_lstm_regg(64)
    lstm_r = U.build_lstm_regg(64)
    lstm_res = U.build_lstm_regg(64)
    lstm_cal = U.build_lstm_regg(64)
    lstm_p_temp = U.build_lstm_regg(64)
    lstm_r_temp = U.build_lstm_regg(64)
    lstm_res_temp = U.build_lstm_regg(64)
    lstm_cal_temp = U.build_lstm_regg(64)
    lstm_p_cons = U.build_lstm_regg(64)
    lstm_r_cons = U.build_lstm_regg(64)
    lstm_res_cons = U.build_lstm_regg(64)
    lstm_cal_cons = U.build_lstm_regg(64)

    cnn_p_moves = U.build_pretrained_cnn()
    cnn_r_moves = U.build_pretrained_cnn()
    cnn_res_moves = U.build_pretrained_cnn()
    cnn_cal_moves = U.build_pretrained_cnn()
    cnn_p_LMouse = U.build_pretrained_cnn()
    cnn_r_LMouse = U.build_pretrained_cnn()
    cnn_res_LMouse = U.build_pretrained_cnn()
    cnn_cal_LMouse = U.build_pretrained_cnn()
    cnn_p_WMouse = U.build_pretrained_cnn()
    cnn_r_WMouse = U.build_pretrained_cnn()
    cnn_res_WMouse = U.build_pretrained_cnn()
    cnn_cal_WMouse = U.build_pretrained_cnn()
    cnn_p_RMouse = U.build_pretrained_cnn()
    cnn_r_RMouse = U.build_pretrained_cnn()
    cnn_res_RMouse = U.build_pretrained_cnn()
    cnn_cal_RMouse = U.build_pretrained_cnn()
    # TRAIN:
    for matcher in np.array(Y[['matcher', 'P', 'R', 'Res', 'Cal', 'P_bin', 'R_bin', 'Res_bin', 'Cal_bin']])[:].copy():
        consensus_seqs[matcher[0]] = U.bulid_consensus_seq(consensus, match_seqs[matcher[0]])
        # if matcher[0].split('_')[0] in test:
        #     continue
        y_p = np.array(matcher[1]).reshape(1, 1)
        y_r = np.array(matcher[2]).reshape(1, 1)
        y_res = np.array(matcher[3]).reshape(1, 1)
        y_cal = np.array(matcher[4]).reshape(1, 1)

        y_p_bin = U.one_hot(np.array(matcher[5]).reshape(1, 1), 2)
        y_r_bin = U.one_hot(np.array(matcher[6]).reshape(1, 1), 2)
        y_res_bin = U.one_hot(np.array(matcher[7]).reshape(1, 1), 2)
        y_cal_bin = U.one_hot(np.array(matcher[8]).reshape(1, 1), 2)

        # LSTM:
        # CONF:
        x = np.array(conf_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        lstm_p.fit(x, y_p, epochs=conf.epochs)
        lstm_r.fit(x, y_r, epochs=conf.epochs)
        lstm_res.fit(x, y_res, epochs=conf.epochs)
        lstm_cal.fit(x, y_cal, epochs=conf.epochs)

        # TIME:
        x = np.array(time_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        lstm_p_temp.fit(x, y_p, epochs=conf.epochs)
        lstm_r_temp.fit(x, y_r, epochs=conf.epochs)
        lstm_res_temp.fit(x, y_res, epochs=conf.epochs)
        lstm_cal_temp.fit(x, y_cal, epochs=conf.epochs)

        # CONSENSUS:
        x = np.array(consensus_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        lstm_p_cons.fit(x, y_p, epochs=conf.epochs)
        lstm_r_cons.fit(x, y_r, epochs=conf.epochs)
        lstm_res_cons.fit(x, y_res, epochs=conf.epochs)
        lstm_cal_cons.fit(x, y_cal, epochs=conf.epochs)

        # CNN:
        # MOVE:
        x = preprocess_input(np.array([heatmaps['Move'][matcher[0]]]))
        cnn_p_moves.fit(x, y_p_bin, epochs=conf.epochs)
        cnn_r_moves.fit(x, y_r_bin, epochs=conf.epochs)
        cnn_res_moves.fit(x, y_res_bin, epochs=conf.epochs)
        cnn_cal_moves.fit(x, y_cal_bin, epochs=conf.epochs)
        # cnn_p_moves.fit(x, y_p, epochs=conf.epochs)
        # cnn_r_moves.fit(x, y_r, epochs=conf.epochs)
        # cnn_res_moves.fit(x, y_res, epochs=conf.epochs)
        # cnn_cal_moves.fit(x, y_cal, epochs=conf.epochs)

        # LMouse:
        x = preprocess_input(np.array([heatmaps['LMouse'][matcher[0]]]))
        cnn_p_LMouse.fit(x, y_p_bin, epochs=conf.epochs)
        cnn_r_LMouse.fit(x, y_r_bin, epochs=conf.epochs)
        cnn_res_LMouse.fit(x, y_res_bin, epochs=conf.epochs)
        cnn_cal_LMouse.fit(x, y_cal_bin, epochs=conf.epochs)

        # WMouse:
        x = preprocess_input(np.array([heatmaps['WMouse'][matcher[0]]]))
        cnn_p_WMouse.fit(x, y_p_bin, epochs=conf.epochs)
        cnn_r_WMouse.fit(x, y_r_bin, epochs=conf.epochs)
        cnn_res_WMouse.fit(x, y_res_bin, epochs=conf.epochs)
        cnn_cal_WMouse.fit(x, y_cal_bin, epochs=conf.epochs)

        # RMouse:
        x = preprocess_input(np.array([heatmaps['RMouse'][matcher[0]]]))
        cnn_p_RMouse.fit(x, y_p_bin, epochs=conf.epochs)
        cnn_r_RMouse.fit(x, y_r_bin, epochs=conf.epochs)
        cnn_res_RMouse.fit(x, y_res_bin, epochs=conf.epochs)
        cnn_cal_RMouse.fit(x, y_cal_bin, epochs=conf.epochs)

    # TEST:
    for matcher in np.array(Y[:])[:].copy():
        curr_feat = features[matcher[0]].copy()
        temp = []
        # LSTM:
        # CONF:
        x = np.array(conf_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        temp += [float(lstm_p.predict(x)[0]), float(lstm_r.predict(x)[0]),
                float(lstm_res.predict(x)[0]), float(lstm_cal.predict(x)[0])]

        # TIME:
        x = np.array(time_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        temp += [float(lstm_p_temp.predict(x)[0]), float(lstm_r_temp.predict(x)[0]),
                float(lstm_res_temp.predict(x)[0]), float(lstm_cal_temp.predict(x)[0])]

        # CONSENSUS:
        x = np.array(consensus_seqs[matcher[0]])
        x = x.reshape(1, x.shape[0], 1)
        temp += [float(lstm_p_cons.predict(x)[0]), float(lstm_r_cons.predict(x)[0]),
                float(lstm_res_cons.predict(x)[0]), float(lstm_cal_cons.predict(x)[0])]

        # CNN:
        # MOVE:
        x = preprocess_input(np.array([heatmaps['Move'][matcher[0]]]))
        # temp += [float(np.argmax(cnn_p_moves.predict(x))), float(np.argmax(cnn_r_moves.predict(x))),
        #         float(np.argmax(cnn_res_moves.predict(x))), float(np.argmax(cnn_cal_moves.predict(x)))]
        # print(cnn_p_moves.predict(x))
        temp += [float(cnn_p_moves.predict(x)[0][1]), float(cnn_r_moves.predict(x)[0][1]),
                float(cnn_res_moves.predict(x)[0][1]), float(cnn_cal_moves.predict(x)[0][1])]


        # LMouse:
        x = preprocess_input(np.array([heatmaps['LMouse'][matcher[0]]]))
        temp += [float(cnn_p_LMouse.predict(x)[0][1]), float(cnn_r_LMouse.predict(x)[0][1]),
                float(cnn_res_LMouse.predict(x)[0][1]), float(cnn_cal_LMouse.predict(x)[0][1])]

        # WMouse:
        x = preprocess_input(np.array([heatmaps['WMouse'][matcher[0]]]))
        temp += [float(cnn_p_WMouse.predict(x)[0][1]), float(cnn_r_WMouse.predict(x)[0][1]),
                float(cnn_res_WMouse.predict(x)[0][1]), float(cnn_cal_WMouse.predict(x)[0][1])]

        # RMouse:
        x = preprocess_input(np.array([heatmaps['RMouse'][matcher[0]]]))
        temp += [float(cnn_p_RMouse.predict(x)[0][1]), float(cnn_r_RMouse.predict(x)[0][1]),
                float(cnn_res_RMouse.predict(x)[0][1]), float(cnn_cal_RMouse.predict(x)[0][1])]

        features[matcher[0]] = np.nan_to_num(np.append(curr_feat, temp))

    X = E.features2pandas(features, True)
    x_test = np.array(X[X['matcher'].isin(test)].drop('matcher', axis=1))
    x_train = np.array(X[~X['matcher'].isin(test)].drop('matcher', axis=1))
    predictions = Y[Y['matcher'].isin(test)][['matcher', 'P_bin', 'R_bin', 'Res_bin', 'Cal_bin']]

    y_train = np.array(Y[~Y['matcher'].isin(test)]['P_bin'])
    y_test = np.array(Y[Y['matcher'].isin(test)]['P_bin'])
    clf = SVC(probability=True, kernel='linear')
    if U.check_labels(y_train):
        clf.fit(x_train, y_train)
        print('Precision:')
        print(classification_report(y_test, clf.predict(x_test)))
        predictions['P_bin_pred'] = clf.predict(x_test)
    else:
        predictions['P_bin_pred'] = y_train[0]

    y_train = np.array(Y[~Y['matcher'].isin(test)]['R_bin'])
    y_test = np.array(Y[Y['matcher'].isin(test)]['R_bin'])
    clf = SVC(probability=True, kernel='linear')
    if U.check_labels(y_train):
        clf.fit(x_train, y_train)
        print('Recall:')
        print(classification_report(y_test, clf.predict(x_test)))
        predictions['R_bin_pred'] = clf.predict(x_test)
    else:
        print(y_train[0])
        predictions['R_bin_pred'] = y_train[0]

    y_train = np.array(Y[~Y['matcher'].isin(test)]['Res_bin'])
    y_test = np.array(Y[Y['matcher'].isin(test)]['Res_bin'])
    clf = SVC(probability=True, kernel='linear')
    if U.check_labels(y_train):
        clf.fit(x_train, y_train)
        print('Res:')
        print(classification_report(y_test, clf.predict(x_test)))
        predictions['Res_bin_pred'] = clf.predict(x_test)
    else:
        predictions['Res_bin_pred'] = y_train[0]

    y_train = np.array(Y[~Y['matcher'].isin(test)]['Cal_bin'])
    y_test = np.array(Y[Y['matcher'].isin(test)]['Cal_bin'])
    clf = SVC(probability=True, kernel='linear')
    if U.check_labels(y_train):
        clf.fit(x_train, y_train)
        print('Cal:')
        print(classification_report(y_test, clf.predict(x_test)))
        predictions['Cal_bin_pred'] = clf.predict(x_test)
    else:
        predictions['Cal_bin_pred'] = y_train[0]
    # temp = pd.merge(X[X['matcher'].isin(test)], predictions, on='matcher', how='inner')
    res = pd.concat([res, predictions], ignore_index=True).drop_duplicates().reset_index(drop=True)
    i += 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st
if not os.path.exists(folder):
    os.makedirs(folder)
res.sort_values(by='matcher', ascending=True).to_csv(folder + '/full_results.csv', index=False)
os.rename('./quality.csv', folder + '/quality.csv')
os.rename('./features.csv', folder + '/features.csv')
