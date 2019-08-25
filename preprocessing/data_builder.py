#!/usr/bin/python
# -*- coding: utf-8 -*-
from argument_BERT.preprocessing import feature_extractor
import numpy as np
import pandas as pd


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


def generate_data(arg1,
                  arg2=None,
                  originalArg1=None,
                  originalArg2=None,
                  fullText1=None):
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
    if fullText1 is not None:
        data['fullText1'] = fullText1

    df = pd.DataFrame(data)
    
    if fullText1 is not None:
        if arg2 is not None:
            temp = df.apply(lambda row:
                            generate_position_features(row['arg1'],
                                                       row['arg2'],
                                                       row['fullText1']))
            temp = pd.DataFrame(temp.tolist(), columns=['positionDiff',
                                                        'positArg1',
                                                        'positArg2',
                                                        'sentenceDiff',
                                                        'sen1',
                                                        'sen2'])
            df['positionDiff'] = temp.loc[:,'positionDiff']
            df['positArg1'] = temp.loc[:,'positArg1']
            df['positArg2'] = temp.loc[:,'positArg2']
            df['sentenceDiff'] = temp.loc[:,'sentenceDiff']
            df['sen1'] = temp.loc[:,'sen1']
            df['sen2'] = temp.loc[:,'sen2']
        else:
            df['positArg1'] = df.apply(lambda row:
                                       generate_position_features(
                                           row['arg1'], None,
                                           row['fullText1']))
    
    return df


def generate_position_features(arg1, arg2=None, fullText=None):
    if fullTex is None:
        return None
    orig_len = len(fullText)
    positArg1 = fullText.find(arg1)
    if arg2 is not None:
        positArg2 = fullText.find(arg2)
    sens = sent_tokenize(fullText)
    for sentence in sens:
        if arg1 in sentence:
            sen1 = sens.index(sentence)
        if arg2 is not None:
            if arg2 in sentence:
                sen2 = sens.index(sentence)
    if arg2 is not None:
        posit = abs((positArg1 - positArg2) / orig_len)
        senit = abs(sen1-sen2) / len(sens)
        data = {'positionDiff': posit,
                'positArg1': positArg1 / orig_len,
                'positArg2': positArg2 / orig_len,
                'sentenceDiff': senit,
                'sen1': sen1 / len(sens),
                'sen2': sen2 / len(sens)}
        return [data['positionDiff'],
                data['positArg1'],
                data['positArg2'],
                data['sentenceDiff'],
                data['sen1'],
                data['sen2']]
    else:
        data = {'positArg1': positArg1 / orig_len}
        return data['positArg1']


def generate_data_with_features(arg1,
                                arg2=None,
                                originalArg1=None,
                                originalArg2=None,
                                fullText=None,
                                bert_embedding=None):
    data = generate_data(arg1, arg2, originalArg1, originalArg2, fullText)
    has_2 = arg2 is not None
    data = add_features(data, has_2, bert_emb=bert_embedding)
    return data


def remove_nongenerable_features(data, bert_embedding):
    basic_text = """Because this is nice! A so welcomed sentence.
                     I hope it will work. As this is a pretty sentence.
                     Global warming harms the people."""
    generable_data = generate_data(['Because this is nice!',
                                    'A so welcomed sentence',
                                    'I hope it will work'],
                                   ['I hope it will work',
                                    'As this is a pretty sentence',
                                    'Global warming harms the people'],
                                   None, None,
                                   [basic_text, basic_text, basic_text])
    generable_data = add_features(generable_data,
                                  bert_emb=bert_embedding)

    for key in data.keys():
        if key not in generable_data.keys():
            if key not in ['label', 'originalLabel']:
                data = data.drop(key, axis=1)

    return data
