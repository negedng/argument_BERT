#!/usr/bin/python
# -*- coding: utf-8 -*-
from argument_BERT.preprocessing import data_loader, preparator
from argument_BERT.preprocessing import model_builder
from argument_BERT.preprocessing import data_builder, data_saver
import numpy as np
import pandas as pd
from bert_embedding import BertEmbedding
from keras.models import load_model
from datetime import datetime
from nltk.tokenize import sent_tokenize


def proposition_identification(text):
    """Find argument propositions in text"""
    sents = sent_tokenize(text)

    props = []
    for i in range(len(sents)):
        next_prop = {'id': ('T' + str(i)),
                     'text': sents[i],
                     'relations': []}
        props.append(next_prop)
    return props


def proposition_type(props, full_text, model_path, verbose=1):
    """Identifies proposition types"""
    texts = [prop['text'] for prop in props]
    ADUs, confs = predict_type(texts, full_text, model_path, verbose=verbose)

    for i in range(len(props)):
        ADU_dict = {0: 'premise', 1: 'claim', 2: 'conclusion'}
        props[i]['ADU'] = {'type': ADU_dict[ADUs[i]],
                           'confidence': str(confs[i])}
    return props


def proposition_position(props, text):
    """Add proposition start and end prefixes"""
    for i in range(len(props)):
        prop_text = props[i]['text']
        s = text.find(prop_text)
        f = s + len(prop_text) - 1
        props[i]['textPosition'] = {'start': str(s),
                                    'end': str(f)}
    return props


def relation_detection(props, text, model_path, verbose=1):
    """Identifies relations between propositions"""
    arg1s = []
    arg1sID = []
    arg2s = []
    arg2sID = []
    for i in range(len(props)):
        for j in range(i+1, len(props)):
            arg1s.append(props[i]['text'])
            arg2s.append(props[j]['text'])
            arg1sID.append(i)
            arg2sID.append(j)
    preds, confs = predict_relation(arg1s, arg2s, text, model_path,
                                    verbose=verbose)

    for i in range(len(preds)):
        if preds[i] != 0:
            arg1ID = arg1sID[i]
            arg2ID = arg2sID[i]

            relID = 'RT' + str(arg1ID) + '-T' + str(arg2ID)
            tyB = preds[i] - 1
            tyStr = 'Default Inference' if tyB == 0 else 'Default Conflict'
            relation = {'id': relID,
                        'typeBinary': str(tyB),
                        'type': tyStr,
                        'partnerID': props[arg2ID]['id'],
                        'confidence': str(confs[i])}
            props[arg1ID]['relations'].append(relation)
    return props


def predict_type(list_of_props, full_text, model_path, verbose=1):
    """Model prediction for proposition types"""
    chances = predictor(model_path, list_of_props,
                        ADU=True, fullText=full_text,
                        verbose=verbose)
    predictions = np.argmax(chances, axis=1)
    confidences = np.max(chances, axis=1)
    return predictions, confidences


def predict_relation(arg1, arg2, full_text, model_path, verbose=1):
    """Model prediction for relation types"""
    chances = predictor(model_path, arg1, arg2,
                        ADU=False, fullText=full_text,
                        verbose=verbose)
    predictions = np.argmax(chances, axis=1)
    confidences = np.max(chances, axis=1)
    return predictions, confidences


def predictor(model_path,
              arg1,
              arg2=None,
              originalArg1=None,
              originalArg2=None,
              ADU=False,
              verbose=1,
              fullText=None):
    """Generates model readable data from propositions to predict"""
    if verbose > 0:
        print('Start loading resources...')
    model = load_model(model_path)
    bert_embedding = BertEmbedding(model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased',
                                   max_seq_length=35)
    if verbose > 0:
        print('Generate prediction...')
    fulls = [fullText.replace('\n', ' ')] * len(arg1)
    data = data_builder.generate_data_with_features(arg1, arg2, originalArg1,
                                                    originalArg2, fulls,
                                                    bert_embedding)
    if not ADU:
        features = model_builder.select_FFNN_features(
            data, shared_feature_list=None, original_bert=True)
        prediction = model.predict(features)
        if verbose > 0:
            print(prediction)
        return prediction
    else:
        features = model_builder.select_FFNN_features(
            data, shared_feature_list=None, original_bert=True, has_2=False)
        prediction = model.predict(features)
        if verbose > 0:
            print(prediction)
        return prediction


def argumentor(text, adu_model, relation_model, corpus_name="NoName",
               out_filename="out.json", verbose=1):
    """Proposition identification, proposition type classification
        and relation detection pipeline.
    Input:
        text: text of the argumentation
        adu_model: path of the proposition type predictor model
        relation_model: path of the relation detector model
        corpus_name: name of the file's original corpus
        out_filename: file to store the results. .json or .xml
        verbose: more than 0 for follow-up texts
    """

    props = proposition_identification(text)

    props = proposition_position(props, text)
    props = proposition_type(props, text, adu_model, verbose=verbose)
    props = relation_detection(props, text, relation_model, verbose=verbose)

    data = {'propositions': props,
            'originalText': text,
            'corpus': corpus_name}

    data_saver.save2file(data, out_filename)
