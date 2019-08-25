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

 
def proposition_identification(text):
    sents = sent_tokenize(text)
  
    props = []
    for i in range(len(sents)):
        next_prop = {'id' : ('T' + str(i)),
                     'text':sents[i],
                     'relations': []}
        props.append(next_prop)
    return props


def proposition_type(props, full_text, model_path):
    texts = [ prop['text'] for prop in props]
    ADUs, confs = predict_type(texts, full_text, model_path)
    
    for i in range(len(props)):
        props[i]['ADU'] = {'type': ADUs[i],
                           'confidence': confs[i]}
    return props


def proposition_position(props, text):
    for i in range(len(props)):
        prop_text = props[i]['text']
        s = text.find(prop_text)
        f = s + len(prop_text) - 1
        props[i]['textPosition'] = {'start':s,
                                'end':f}
    return props


def relation_detection(props, text, model_path):
    arg1s = []
    arg1sID = []
    arg2s = []
    arg2sID = []
    for i in range(len(props)):
        for j in range(i+1,len(props)):
            arg1s.append(props[i])
            arg2s.append(props[j])
            arg1sID.append(i)
            arg2sID.append(j)
    preds, confs = predict_relation(arg1s, arg2s, text, model_path)
    
    for i in range(len(preds)):
        if preds[i] != 0:
            arg1ID = arg1sID[i]
            arg2ID = arg2sID[i]
            
            relID = 'RT' + str(arg1ID) + '-T' + str(arg2ID)
            tyB = preds[i] - 1
            tyStr = 'Default Inference' if tyB==0 else 'Default Conflict'
            relation = {'id': relID,
                        'typeBinary': tyB,
                        'type': tyStr,
                        'partnerID': props[arg2ID]['id']}
            props[arg1ID]['relations'].append(relation)   
    return props
            

def predict_type(list_of_props, full_text, model_path):
    chances = predictor(model_path, list_of_props,
                        ADU=True, fullText=full_text)
    predictions = np.argmax(chances, axis=1)
    confidences = np.max(chances, axis=1)
    return predictions, confidences


def predict_relation(arg1, arg2, full_text, model_path):
    chances = predictor(model_path, arg1, arg2,
                        ADU=False, fullText=full_text)
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
    if verbose > 0:
        print('Start loading resources...')
    model = load_model(model_path)
    bert_embedding = BertEmbedding(model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased',
                                   max_seq_length=35)
    if verbose > 0:
        print('Generate prediction...')
    fulls = [fullText] * len(arg1)
    data = data_builder.generate_data_with_features(arg1, arg2, originalArg1,
                                                    originalArg2, fullText,
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
               out_filename="out.json"):
    props = proposition_identification(text)
    
    props = proposition_position(props, text)
    props = proposition_type(props, text, adu_model)
    props = relation_detection(props, text, relation_model)
    
    data = {'propositions': props,
            'originalText': text,
            'corpus': corpus_name}

    data_saver.save2file(data, out_filename)
