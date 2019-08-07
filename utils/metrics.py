#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_curve


def apply_threshold(A, th=0.5):
    """If a<th -> a=0, else a=1"""

    A = np.array(A)
    A[A < th] = 0
    A[A >= th] = 1
    return A


def encode_one_hot(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers)
    to binary class matrix."""

    return to_categorical(y, num_classes, dtype)


def decode_one_hot(y):
    """Returns with the max class"""

    return np.argmax(y, -1)


def classification_report_with_threshold(y_true, y_pred, th=None):
    """Decode one-hot classification with max or threshold
    and run sklearn classification_report"""

    y_true = decode_one_hot(y_true)
    if th is not None:
        print('Apply threshold')
        y_pred = apply_threshold(y_pred, th)
    y_pred = decode_one_hot(y_pred)
    rep = classification_report(y_true, y_pred)
    rep = 'Threshold: ' + str(th) + '''

''' + rep
    return rep


def draw_ROC_curve(y_true, y_pred, verbose=1):
    """Draws ROC curve for the data,
    returns dictionaries of fpr, tpr and thr"""

    fpr = dict()
    tpr = dict()
    thr = dict()
    try:
        no_classes = np.shape(y_true)[1]
        for i in range(no_classes):
            (fpr[i], tpr[i], thr[i]) = roc_curve(y_true[:, i], y_pred[:, i])
    except IndexError:
        (fpr[0], tpr[0], thr[0]) = roc_curve(y_true, y_pred)
    plt.figure()

    if verbose > 0:
        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], label='ROC curve for class ' + str(i))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend()
        plt.show()
    return (fpr, tpr, thr)


def related_unrelated_report(model,
                             features_test,
                             y_test,
                             target_names=None,
                             ):
    prediction = model.predict(features_test)

    numberOfClasses = y_test.shape[1]

    position = np.argmax(prediction, axis=-1)
    y_pred = np.identity(numberOfClasses)[position]

    if target_names is None:
        if numberOfClasses == 2:
            target_names = ['nonrelated', 'related']
        else:
            if numberOfClasses == 3:
                target_names = ['nonrelated', 'support', 'attack']
            else:
                if numberOfClasses == 5:
                    target_names = ['nonrelated', 'support', 'attack',
                                    'attacked', 'supported']
                else:
                    target_names = None

    print(classification_report(y_test, y_pred,
                                target_names=target_names))


def adu_report(model, features_test, y_test):
    related_unrelated_report(model, features_test, y_test, ['premise',
                             'claim', 'conclusion'])
