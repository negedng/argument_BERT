#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle


def input_output_split(dataset):
    """Prepare dataset to training. Starts feature extracting.
    Returns x_data, y_data"""

    y_data = dataset['label'].to_numpy()
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)

    return (x_data, y_data)


def change_labels(dataset,
                  attack=False,
                  bidirect='abs',
                  save_original=True,
                  ):
    """Changes the labels depending on the classification task
     Input:
        dataset: pandas dataframe containing the data
        attack: if true, the attack label will remain in the dataset,
            default is false
        bidirect: abs - absolute value,
                  zero - negatives zero,
                  skip - keeping original
        save_original: save the original label into an other row
    """

    if save_original:
        dataset['originalLabel'] = dataset['label']

    if not attack:
        dataset.loc[dataset.label == 2, 'label'] = 1
        dataset.loc[dataset.label == -2, 'label'] = -1

    if bidirect == 'abs':
        dataset.loc[dataset.label == -1, 'label'] = 1
        dataset.loc[dataset.label == -2, 'label'] = 2
    else:
        if bidirect == 'zero':
            dataset.loc[dataset.label == -1, 'label'] = 0
            dataset.loc[dataset.label == -2, 'label'] = 0

    return dataset


def train_test_split(dataset, split_ratio=0.1):
    """Splits the data into training and testing, input and output
     Input:
        dataset: pandas dataframe containing the data
        split_ratio: ratio of the test data
    """

    (train_data, test_data) = sk_train_test_split(dataset,
                                                  test_size=split_ratio,
                                                  random_state=42)

    (x_train, y_train) = input_output_split(train_data)
    (x_test, y_test) = input_output_split(test_data)

    return (x_train, x_test, y_train, y_test)


def balance_dataset(dataset, balance_ratio):
    """Reduce the number of unrelated data examples to match the related ones.
    Input:
        dataset: pandas dataframe containing the data
        balance_ratio: precentage of the balancing: 0.5 = equal balancing
    """

    RELATION_RATIO = balance_ratio

    labelMatrix = dataset['label'].to_numpy()
    numberOfRelations = np.count_nonzero(labelMatrix)
    relationRatio = numberOfRelations / len(dataset)

    if relationRatio < RELATION_RATIO:
        dataset['labelAbs'] = dataset['label'].abs()

        print('-----DATA IS UNBALANCED CURRENT SIZE: '
              + str(len(dataset)) + ' CLASS RATIO: ' + str(relationRatio)
              + ' ... BALANCING DATA')

        shuffled = sk_shuffle(dataset)

        orderedDataset = shuffled.sort_values(by=['labelAbs'],
                                              ascending=False)
        cutOff = int(1 / RELATION_RATIO * numberOfRelations)

        balanced = sk_shuffle(orderedDataset.head(cutOff))
        balanced = balanced.drop('labelAbs', axis=1)

        print('-----BALANCED DATASET WITH SIZE: ' + str(len(balanced)))
        return balanced
    else:

        print('-----DATASET IS ALREADY BALANCED - CLASS RATIO: '
              + str(relationRatio) + '-----')

        return dataset


def lower_texts(dataset, columns=['arg1', 'arg2', 'originalArg1',
                'originalArg2']):
    """Lower texts in columns"""

    for column in columns:
        dataset[column] = dataset[column].apply(lambda row: row.lower())
    return dataset


def reset_labels(dataset):
    """Set labels to the original labels stored in originalLabel"""
    dataset['label'] = dataset['originalLabel']
    return dataset
