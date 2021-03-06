import keras
from keras.models import Sequential, Model
from keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, Flatten, Permute, \
    Dropout, Lambda, Reshape, Embedding, GlobalAveragePooling2D
from keras.utils import to_categorical
import numpy as np
# from keras.applications import VGG19
# from keras.applications.vgg19 import preprocess_input
# from keras.applications import VGG16
# from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import load_img
import tensorflow as tf
from keras import backend as K
# import shap
import matplotlib.pyplot as plt

pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(37, 45, 3), classes=2)
graph = tf.get_default_graph()


def load_vgg():
    global pretrained
    pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(37, 45, 3), classes=2)
    global graph
    graph = tf.get_default_graph()


def build_lstm_classify(first):
    model = Sequential()
    model.add(LSTM(first, input_shape=(None, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
    # print(model.summary())
    return model


def build_lstm_regg(first):
    model = Sequential()
    model.add(LSTM(first, input_shape=(None, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["acc"])
    # print(model.summary())
    return model


def build_pretrained_cnn():
    global graph
    with graph.as_default():
        for layer in pretrained.layers[:5]:
            layer.trainable = False

        # Adding custom Layers
        x = pretrained.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        # predictions = Dense(1, activation='relu')(x)
        model = Model(inputs=pretrained.input, outputs=predictions)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    # print(model.summary())
    return model


def bulid_consensus(matches):
    consensus = {}
    for m in matches:
        for c in matches[m]:
            if c[0] not in consensus:
                consensus[c[0]] = 0
            consensus[c[0]] += 1
    return consensus


def bulid_consensus_seq(consensus, match_seq):
    consensus_seq = []
    for m in match_seq:
        val = 0
        if m in consensus:
            val = consensus[m]
        consensus_seq += [val, ]
    return consensus_seq


def check_labels(y):
    if len(list(np.unique(y))) == 1:
        return False
    return True


def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1).astype(int)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets


def check_importance(X, model):
    # DF, based on which importance is checked
    X_importance = X

    # Explain model predictions using shap library:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_importance)
    shap.summary_plot(shap_values, X_importance)
    return True