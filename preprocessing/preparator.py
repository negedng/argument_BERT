from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np
import pandas as pd

def prepare_data_for_training(dataset, model_type=None):
    """Prepare dataset to training. Starts feature extracting.
    Returns x_data, y_data"""
    y_data = dataset['label'].to_numpy()
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)

    return x_data, y_data
	
def change_labels(dataset, attack=False, bidirect=True):
    """changes the labels depending on the classification task
        parameter:
        dataset: pandas dataframe containing the data
        attack: if true, the attack label will remain in the dataset, default is false
        bidirect: allow bidirectional relation instead of one directional
    """

    if not attack:
        dataset.loc[dataset.label == 2, 'label'] = 1
        dataset.loc[dataset.label == -2, 'label'] = -1

    if bidirect:
        dataset.loc[dataset.label == -1, 'label'] = 1
    else:
        dataset.loc[dataset.label == -1, 'label'] = 0

    return dataset

def train_test_split(dataset, split_ratio=0.1):
    """Splits the data into training and testing, input and output
        parameter:
        dataset: pandas dataframe containing the data
        split_ratio: ratio of the test data
    """

    y_data = dataset.as_matrix(columns=['label'])
    x_data = dataset.drop(['label', 'argumentationID'], axis=1)
    x_train, x_test, y_train, y_test = sk_train_test_split(x_data, y_data, test_size=split_ratio, random_state=42, stratify=y_data)

    return x_train, x_test, y_train, y_test