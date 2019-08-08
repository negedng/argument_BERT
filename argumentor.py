#!/usr/bin/python
# -*- coding: utf-8 -*-
from argument_BERT.preprocessing import data_loader, preparator
from argument_BERT.preprocessing import feature_extractor, model_builder
from argument_BERT.utils import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from keras.preprocessing.sequence import pad_sequences as kp_pad_sequences
from bert_embedding import BertEmbedding
from keras.callbacks import EarlyStopping
from keras.models import load_model
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


def add_features(data, has_2=True, bert_emb=None):
    has_full = 'fullText1' in data.keys()
    (propositionSet, parsedPropositions) = \
        feature_extractor.get_propositions(data)
    (propOriginalSet, parsedPropOriginal) = \
        feature_extractor.get_propositions(data, 'originalArg1')
    if has_full:
        (propFullSet, parsedPropFull) = \
            feature_extractor.get_propositions(data, 'fullText1')

#   data = feature_extractor.add_word_vector_feature(data, propositionSet,
#                                                    parsedPropositions)
#   data = feature_extractor.add_pos_feature(data, propositionSet,
#                                                  parsedPropositions)

    data = feature_extractor.add_keyword_feature(data, has_2=has_2)
    data = feature_extractor.add_token_feature(data, propositionSet,
                                               parsedPropositions, has_2=has_2)
    data = feature_extractor.add_shared_words_feature(
        data,
        propositionSet,
        parsedPropositions,
        'arg',
        'nouns',
        0,
        has_2=has_2,
        )
    data = feature_extractor.add_shared_words_feature(
        data,
        propositionSet,
        parsedPropositions,
        'arg',
        'verbs',
        0,
        has_2=has_2,
        )
    data = feature_extractor.add_shared_words_feature(
        data,
        propositionSet,
        parsedPropositions,
        'arg',
        'words',
        3,
        True,
        has_2=has_2,
        )
    data = feature_extractor.add_shared_words_feature(
        data,
        propOriginalSet,
        parsedPropOriginal,
        'originalArg',
        'nouns',
        0,
        has_2=has_2,
        )
    data = feature_extractor.add_shared_words_feature(
        data,
        propOriginalSet,
        parsedPropOriginal,
        'originalArg',
        'verbs',
        0,
        has_2=has_2,
        )
    data = feature_extractor.add_shared_words_feature(
        data,
        propOriginalSet,
        parsedPropOriginal,
        'originalArg',
        'words',
        3,
        True,
        has_2=has_2,
        )

#    data = feature_extractor.add_shared_words_feature(data, propositionSet,
#                                                      parsedPropositions,
#                                                      'arg', 'words', 3, True,
#                                                      True, propFullSet,
#                                                      parsedPropFull,
#                                                      has_2=has_2)

    data = feature_extractor.add_same_sentence_feature(data, has_2=has_2)
    data = feature_extractor.add_bert_embeddings(data, True, False, True,
                                                 bert_embedding=bert_emb,
                                                 has_2=has_2,)
    data = feature_extractor.add_sentiment_scores(data, 'arg',
                                                  has_2=has_2)
    data = feature_extractor.add_sentiment_scores(data, 'originalArg',
                                                  has_2=has_2)
    if has_full:
        data = feature_extractor.add_sentiment_scores(data, 'fullText',
                                                      has_2=False)
    return data


def reset_labels(dataset):
    dataset['label'] = dataset['originalLabel']
    return dataset


def trainer(directory,
            ADU=False,
            save=True,
            save_dir='/content/models/',
            train_generable=False,
            support_attack=True,):
    bert_embedding = BertEmbedding(model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased',
                                   max_seq_length=35)

    train_data = load_data(directory, ADU=ADU, attack=support_attack)
    train_data = add_features(train_data, has_2=not ADU,
                              bert_emb=bert_embedding)
    if train_generable:
        train_data = remove_nongenerable_features(train_data,
                                                  bert_embedding)
    (train_data, test_data) = sk_train_test_split(train_data,
                                                  test_size=0.10)
    print 'Train-test split:' + str(len(train_data)) + ' ' \
        + str(len(test_data))

    (x_data, y_data) = preparator.input_output_split(train_data)
    (x_test, y_test) = preparator.input_output_split(test_data)
    print 'X-Y data ready: ' + str(len(x_data)) + ' ' + str(len(x_test))

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


def generate_data(arg1,
                  arg2=None,
                  originalArg1=None,
                  originalArg2=None,):
    argumentID = [0] * len(arg1)
    data = {'arg1': arg1, 'argumentationID': argumentID}

    if arg2 is not None:
        data['arg2'] = arg2

    if originalArg1 is not None:
        data['originalArg1'] = originalArg1
    else:
        data['originalArg1'] = arg1

    if originalArg2 is not None:
        data['originalArg2'] = originalArg2
    else:
        if arg2 is not None:
            data['originalArg2'] = arg2

    df = pd.DataFrame(data)
    return df


def generate_data_with_features(arg1,
                                arg2=None,
                                originalArg1=None,
                                originalArg2=None,
                                bert_embedding=None,):
    data = generate_data(arg1, arg2, originalArg1, originalArg2)
    has_2 = arg2 is not None
    data = add_features(data, has_2, bert_emb=bert_embedding)
    return data


def remove_nongenerable_features(data, bert_embedding):
    generable_data = generate_data(['Because this is nice!',
                                    'A so welcomed sentence',
                                    'I hope it will work'],
                                   ['I hope it will work',
                                    'as this is a pretty sentence',
                                    'Global warming harms the people'])
    generable_data = add_features(generable_data,
                                  bert_emb=bert_embedding)

    for key in data.keys():
        if key not in generable_data.keys():
            if key not in ['label', 'originalLabel']:
                data = data.drop(key, axis=1)

    return data


def predictor(model_path,
              arg1,
              arg2=None,
              originalArg1=None,
              originalArg2=None,
              ADU=False,
              verbose=1,):
    if verbose > 0:
        print 'Start loading resources...'
    model = load_model(model_path)
    bert_embedding = BertEmbedding(model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased',
                                   max_seq_length=35)
    if verbose > 0:
        print 'Generate prediction...'
    data = generate_data_with_features(arg1, arg2, originalArg1,
                                       originalArg2, bert_embedding)
    if not ADU:
        features = model_builder.select_FFNN_features(
            data, shared_feature_list=None, original_bert=True)
        prediction = model.predict(features)
        if verbose > 0:
            print prediction
        return prediction
    else:
        features = model_builder.select_FFNN_features(
            data, shared_feature_list=None, original_bert=True, has_2=False)
        prediction = model.predict(features)
        if verbose > 0:
            print prediction
        return prediction
