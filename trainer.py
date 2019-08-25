#!/usr/bin/python
# -*- coding: utf-8 -*-
from argument_BERT.preprocessing import preparator
from argument_BERT.preprocessing import model_builder
from argument_BERT.preprocessing import data_builder, data_loader
from argument_BERT.utils import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from bert_embedding import BertEmbedding
from keras.callbacks import EarlyStopping
from datetime import datetime


def load_data(directory,
              rst_files=True,
              attack=True,
              bidirect='abs',
              balance_ratio=0.5,
              ADU=False,
              ):

    train_data = data_loader.load_from_directory(directory, rst_files, ADU)
    if not ADU:
        train_data = preparator.change_labels(train_data,
                                              attack=attack,
                                              bidirect=bidirect)
    train_data = preparator.balance_dataset(train_data, balance_ratio)

    return train_data


def trainer(directory,
            ADU=False,
            save=True,
            save_dir='/content/models/',
            train_generable=False,
            support_attack=True,
            rst_files=True,
            verbose=0):
    bert_embedding = BertEmbedding(model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased',
                                   max_seq_length=35)

    train_data = load_data(directory, ADU=ADU, attack=support_attack,
                           rst_files=rst_files)
    train_data = data_builder.add_features(train_data, has_2=not ADU,
                                           bert_emb=bert_embedding)
    if train_generable:
        train_data = data_builder.remove_nongenerable_features(train_data,
                                                               bert_embedding,
                                                               ADU)
    if verbose>0:
        print('Feature list:')
        print(list(train_data.keys()))
    (train_data, test_data) = sk_train_test_split(train_data,
                                                  test_size=0.10)
    if verbose>0:
        print('Train-test split:' + str(len(train_data)) + ' '
              + str(len(test_data)))

    (x_data, y_data) = preparator.input_output_split(train_data)
    (x_test, y_test) = preparator.input_output_split(test_data)
    if verbose>0:
        print('X-Y data ready: ' + str(len(x_data)) + ' ' + str(len(x_test)))

    es = EarlyStopping('val_loss', patience=150,
                       restore_best_weights=True)

    if not ADU:
        features = model_builder.select_FFNN_features(x_data,
                                                      shared_feature_list=None,
                                                      original_bert=True)
        model = model_builder.build_FFNN(features[-1].shape[1:],
                                         y_data.shape[1],
                                         1, 300, 1, 600, 0.4, 0.10, True, True,
                                         1, 300, optimizer='rmsprop',
                                         activation='sigmoid',)
        history = model.fit(
            features,
            y_data,
            validation_split=0.05,
            epochs=5000,
            batch_size=5000,
            verbose=0,
            callbacks=[es],
            )

        test_features = model_builder.select_FFNN_features(
            x_test, shared_feature_list=None, original_bert=True)
        metrics.related_unrelated_report(model, test_features, y_test)
    else:
        features = model_builder.select_FFNN_features(
            x_data, shared_feature_list=None, original_bert=True, has_2=False)
        model = model_builder.build_FFNN(
            features[-1].shape[1:],
            y_data.shape[1],
            1,
            300,
            1,
            600,
            0.4,
            0.10,
            True,
            True,
            1,
            300,
            optimizer='rmsprop',
            activation='sigmoid',
            has_2=False,
            )
        history = model.fit(
            features,
            y_data,
            validation_split=0.05,
            epochs=5000,
            batch_size=5000,
            verbose=0,
            callbacks=[es],
            )
        test_features = model_builder.select_FFNN_features(
            x_test, shared_feature_list=None, original_bert=True, has_2=False)
        metrics.adu_report(model, test_features, y_test)

    if save:
        filename = 'model_' \
            + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.h5'
        path = save_dir + filename
        model.save(path)
