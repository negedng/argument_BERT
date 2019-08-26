#!/usr/bin/python
# -*- coding: utf-8 -*-
####
#
# Some features based on Tobias Milz's ArguE project.
# See more: https://github.com/Milzi/ArguE
#
####

import nltk
import numpy as np
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences as kp_pad_sequences
from sklearn.preprocessing import LabelBinarizer
import os
import inspect
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import bert_embedding

current_dir = os.path.dirname(inspect.stack()[0][1])
WORD2WEC_EMBEDDING_FILE = \
    '/root/input/GoogleNews-vectors-negative300.bin.gz'
PREMISE_FILE = current_dir + '/premise_indicator.txt'
CLAIM_FILE = current_dir + '/claim_indicator.txt'


def get_propositions(dataset, column='arg1',
                     tokenizer=nltk.tokenize.word_tokenize,
                     only_props=False):
    """Parse propositions
    dataset: the original dataframe
    tokenizer: nltk.tokenize.word_tokenize, bert-embedding.tokenizer, etc.
    Output:
        propositionSet: list of the propositions of the arg1
        parsedPropositions: parsed prop. in the arg1
    """

    propositionSet = list(set(dataset[column]))

    if column[-1] == '1':
        column2 = column[:-1] + '2'
        if column2 in dataset.keys():
            propSet2 = list(set(dataset[column2]))
            propositionSet = propositionSet + propSet2

    propositionSet = list(dict.fromkeys(propositionSet))
    parsedPropositions = list()

    if only_props:
        return (propositionSet, parsedPropositions)

    for proposition in propositionSet:
        words = tokenizer(proposition)
        parsedPropositions.append(nltk.pos_tag(words))

    return (propositionSet, parsedPropositions)


def add_word_vector_feature(dataset,
                            propositionSet,
                            parsedPropositions,
                            word2VecModel=None,
                            pad_no=35,
                            has_2=True,
                            ):
    """Add word2vec feature to the dataset
    """

    if word2VecModel is None:
        KV = gensim.models.KeyedVectors
        word2VecModel = KV.load_word2vec_format(WORD2WEC_EMBEDDING_FILE,
                                                binary=True)

    wordVectorFeature = list()
    feature_vector = np.zeros(300)

    for proposition in parsedPropositions:
        propositionVector = list()
        for word in proposition:
            if word[0] in word2VecModel.vocab:
                feature_vector = word2VecModel[word[0]]

            propositionVector.append(feature_vector)

        wordVectorFeature.append(propositionVector)

    wordVectorFeature = np.array(wordVectorFeature)
    wordVectorFeature = kp_pad_sequences(wordVectorFeature,
                                         maxlen=pad_no, value=0,
                                         padding='post', dtype=float)

    wordVectorFrame = pd.DataFrame({'arg1': propositionSet,
                                   'vector1': wordVectorFeature.tolist()})
    dataset = pd.merge(dataset, wordVectorFrame, on='arg1')
    if has_2:
        wordVectorFrame = wordVectorFrame.rename(columns={'arg1': 'arg2',
                                                          'vector1': 'vector2',
                                                          })
        dataset = pd.merge(dataset, wordVectorFrame, on='arg2')

    return dataset


def add_pos_feature(dataset,
                    propositionSet,
                    parsedPropositions,
                    pad_no=35,
                    has_2=True,
                    ):
    """Add Part-of-Speech features for every proposition"""

    tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
    lb = LabelBinarizer()
    lb.fit(list(tagdict.keys()))

    propositionPOSList = list()

    current = 0
    for proposition in parsedPropositions:

        propositionPOS = get_one_hot_pos(proposition, lb)
        propositionPOSList.append(propositionPOS)

    propositionPOSPadded = kp_pad_sequences(propositionPOSList,
                                            maxlen=pad_no, value=0,
                                            padding='post')

    posFrame = pd.DataFrame({'arg1': propositionSet,
                            'pos1': propositionPOSPadded.tolist()})
    dataset = pd.merge(dataset, posFrame, on='arg1')
    if has_2:
        posFrame = posFrame.rename(columns={'arg1': 'arg2',
                                   'pos1': 'pos2'})
        dataset = pd.merge(dataset, posFrame, on='arg2')

    return dataset


def get_one_hot_pos(parsedProposition, label_binarizer):
    """Get one-hot encoded PoS for the proposition"""

    posVectorList = label_binarizer.transform([word[1] for word in
                                              parsedProposition])
    posVector = np.array(posVectorList)

    return posVector


def add_keyword_feature(dataset, has_2=True):
    """Add premise and claim flag for every proposition"""

    premise_list = read_key_words(PREMISE_FILE)
    claim_list = read_key_words(CLAIM_FILE)

    if has_2:
        all_args = pd.concat([dataset[['arg1', 'originalArg1']],
                              dataset[['arg2', 'originalArg2']
                                       ].rename(columns={
                                           'arg2': 'arg1',
                                           'originalArg2': 'originalArg1'})])
    else:
        all_args = dataset[['arg1', 'originalArg1']]

    keywords = all_args.drop_duplicates().apply(lambda row:
                                                including_keywords_features(
                                                    row['arg1'],
                                                    row['originalArg1'],
                                                    premise_list,
                                                    claim_list), axis=1)
    keywords = pd.DataFrame(keywords.tolist(), columns=['arg1',
                                                        'claimIndicatorArg1',
                                                        'premiseIndicatorArg1'
                                                        ])

    dataset = pd.merge(dataset, keywords, on='arg1')
    if has_2:
        keywords = keywords.rename(columns={
                            'arg1': 'arg2',
                            'claimIndicatorArg1': 'claimIndicatorArg2',
                            'premiseIndicatorArg1': 'premiseIndicatorArg2'
                            })
        dataset = pd.merge(dataset, keywords, on='arg2')

    return dataset


def including_keywords_features(proposition,
                                original,
                                premise_list,
                                claim_list,
                                ):
    """Check if the proposition is a keyword or part of a key phrase
    proposition: to check if keyword
    original: sentence
    premise_list: list of premise keywords
    claim_list: list of claim keywords
    Return:
        [premise, claim] - 1 if sentence contains keyword"""

    positionInSentence = original.find(proposition)

    if positionInSentence < 1:

        claim_indicator = \
            check_claim_indicators(original[:len(proposition)],
                                   claim_list)
        premise_indicator = \
            check_premise_indicators(original[:len(proposition)],
                                     premise_list)
    else:

        wordTokensBefore = \
            nltk.tokenize.word_tokenize(original[:positionInSentence])

        if len(wordTokensBefore) > 1:

            wordsBefore = wordTokensBefore[-2] + wordTokensBefore[-1]
        else:

            wordsBefore = wordTokensBefore[-1]

        extendedSentence = ''.join(wordsBefore) + ' ' + proposition

        claim_indicator = check_claim_indicators(extendedSentence,
                                                 claim_list)
        premise_indicator = check_premise_indicators(extendedSentence,
                                                     premise_list)

    return [proposition, claim_indicator, premise_indicator]


def check_premise_indicators(sentence, premise_list):
    """
    function to detect the presence of argument keywords in a sentence
    :param full sentence:
    :return: 1 if sentence contains keyword
    """

    for indicator in premise_list:
        if re.search(r"\b" + re.escape(indicator.lower()) + r"\b",
                     sentence.lower()):
            return 1
    return 0


def check_claim_indicators(sentence, claim_list):
    """
    function to detect the presence of argument keywords in a sentence
    :param full sentence:
    :return: True if sentence contains keyword
    """

    for indicator in claim_list:
        if re.search(r"\b" + re.escape(indicator.lower()) + r"\b",
                     sentence.lower()):
            return 1
    return 0


def read_key_words(file):
    """Reads list of words in file, one keyword per line"""

    return [line.rstrip('\n') for line in open(file)]


def add_token_feature(dataset,
                      propositionSet,
                      parsedPropositions,
                      has_2=True,
                      ):
    """Add number of propositions in the arguments of the dataset"""

    numberOfTokens = list()

    for i in range(len(propositionSet)):

        numberOfTokens.append([propositionSet[i],
                              len(parsedPropositions[i])])

    tokenDataFrame = pd.DataFrame(data=numberOfTokens,
                                  columns=['proposition', 'tokens'])

    tokenDataFrame = \
        tokenDataFrame.rename(columns={'proposition': 'arg1',
                              'tokens': 'tokensArg1'})

    dataset = pd.merge(dataset, tokenDataFrame, on='arg1')
    if has_2:
        tokenDataFrame = tokenDataFrame.rename(columns={
                                        'arg1': 'arg2',
                                        'tokensArg1': 'tokensArg2'})
        dataset = pd.merge(dataset, tokenDataFrame, on='arg2')

    return dataset


def add_shared_words_feature(dataset,
                             propositionSet,
                             parsedPropositions,
                             key='arg',
                             word_type='nouns',
                             min_word_length=0,
                             stemming=False,
                             fullText=False,
                             fullPropositionSet=None,
                             fullParsedPropositions=None,
                             has_2=True,
                             ):
    """Add binary has shared noun and number of shared nouns to the dataset"""

    if not has_2 and not fullText:
        return dataset
    full = ''
    if fullText:
        full = 'Full'

    if stemming:
        ps = nltk.stem.PorterStemmer()
        stemmed = 'Stem'
    else:
        ps = None
        stemmed = ''
    key1 = key + '1'
    key2 = key + '2'
    word_key = word_type.title()
    if key == 'arg':
        ret_keys = 'shared' + stemmed + word_key + full
        ret_keyn = 'numberOfShared' + stemmed + word_key + full
    else:
        ret_keys = 'originalShared' + stemmed + word_key + full
        ret_keyn = 'originalNumberOfShared' + stemmed + word_key + full

    if word_type == 'nouns':
        pos_tag_list = ['NN']
    else:
        if word_type == 'verbs':
            pos_tag_list = ['VB']
        else:
            pos_tag_list = []
    if not fullText:
        temp = dataset[[key1, key2]
                       ].apply(lambda row:
                               find_shared_words(parsedPropositions[
                                                    propositionSet.index(row[
                                                        key1])],
                                                 parsedPropositions[
                                                     propositionSet.index(row[
                                                         key2])],
                                                 min_length=min_word_length,
                                                 pos_tag_list=pos_tag_list,
                                                 stemming=stemming,
                                                 ps=ps,
                                                 ), axis=1)
        temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                            'numberOfSharedNouns'])
        dataset[ret_keys] = temp.loc[:, 'sharedNouns']
        dataset[ret_keyn] = temp.loc[:, 'numberOfSharedNouns']
    else:
        temp = dataset[[key1, 'fullText1']
                       ].apply(lambda row:
                               find_shared_words(
                                   parsedPropositions[
                                       propositionSet.index(row[key1])],
                                   fullParsedPropositions[
                                       fullPropositionSet.index(
                                           row['fullText1'])],
                                   min_length=min_word_length,
                                   pos_tag_list=pos_tag_list,
                                   stemming=stemming,
                                   ps=ps,
                                   ), axis=1)
        temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                            'numberOfSharedNouns'])
        dataset[ret_keys + '1'] = temp.loc[:, 'sharedNouns']
        dataset[ret_keyn + '1'] = temp.loc[:, 'numberOfSharedNouns']
        if has_2:
            temp = dataset[[key2, 'fullText1']
                           ].apply(lambda row:
                                   find_shared_words(
                                       parsedPropositions[
                                           propositionSet.index(row[key2])],
                                       fullParsedPropositions[
                                           fullPropositionSet.index(
                                               row['fullText1'])],
                                       min_length=min_word_length,
                                       pos_tag_list=pos_tag_list,
                                       stemming=stemming,
                                       ps=ps,
                                       ), axis=1)
            temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns',
                                'numberOfSharedNouns'])
            dataset[ret_keys + '2'] = temp.loc[:, 'sharedNouns']
            dataset[ret_keyn + '2'] = temp.loc[:, 'numberOfSharedNouns']

    return dataset


def find_shared_words(proposition,
                      partner,
                      min_length=0,
                      pos_tag_list=['NN'],
                      stemming=False,
                      ps=None,
                      ):
    """Find shared words between prop and partner
    Input:
        proposition: search key
        partner: search target
        min_length: minimum length of the shared words
        pos_tag_list: PoS tag for collected words, [] for all
        stemming: True for using stemming
        ps: PorterStemmer
    Output:
        sharedWords: binary
        noSharedWords: number of shared words
    """

    has_tag_list = len(pos_tag_list) > 0
    if not stemming:
        arg1Nouns = [word for (word, pos) in proposition
                     if (not has_tag_list or
                         pos in pos_tag_list) and
                     len(word) >= min_length]
        arg2Nouns = [word for (word, pos) in partner
                     if (not has_tag_list or
                         pos in pos_tag_list) and
                     len(word) >= min_length]
    else:
        arg1Nouns = [ps.stem(word) for (word, pos) in proposition
                     if len(word) >= min_length]
        arg2Nouns = [ps.stem(word) for (word, pos) in partner
                     if len(word) >= min_length]

    intersection = set(arg1Nouns).intersection(arg2Nouns)
    shared = 0

    if len(intersection) > 0:
        shared = 1
        return [shared, len(intersection)]
    else:
        return [0.0, 0.0]


def add_same_sentence_feature(dataset, has_2=True):
    """Add binary feature true if the two
       argument has the same original sentence"""

    if not has_2:
        return dataset
    dataset['sameSentence'] = dataset[['originalArg1', 'arg2']
                                      ].apply(lambda row: int(bool(row['arg2']
                                              in row['originalArg1'])), axis=1)

    return dataset


def add_bert_embeddings(dataset,
                        sentence_feature=True,
                        token_feature=True,
                        original_sentence_feature=True,
                        bert_embedding=None,
                        pad_no=35,
                        has_2=True,
                        ):
    """Add bert embeddings to the dataset. Use matching tokenizer!"""

    if bert_embedding is None:
        bert_dataset = 'book_corpus_wiki_en_cased'
        bert_embedding = BertEmbedding(model='bert_12_768_12',
                                       dataset_name=bert_dataset)

    propositionSet, _ = get_propositions(dataset, column='arg1',
                                         only_props=True)

    embeddingSet = bert_embedding(propositionSet,
                                  filter_spec_tokens=False)

    if sentence_feature:
        embs3d = np.array(embeddingSet)[:, 1]
        embs1 = np.array([x[0] for x in embs3d])
        embs2d = np.empty((embs1.shape[0], ), dtype=np.object)
        for i in range(embs1.shape[0]):
            embs2d[i] = embs1[i, :]
        emb_frame = pd.DataFrame(embs2d, columns=['bertArg1'])

        emb_frame['arg1'] = pd.Series(propositionSet,
                                      index=emb_frame.index)

        dataset = pd.merge(dataset, emb_frame, on='arg1')
        if has_2:
            emb_frame = emb_frame.rename(columns={'arg1': 'arg2',
                                                  'bertArg1': 'bertArg2'})
            dataset = pd.merge(dataset, emb_frame, on='arg2')

    if token_feature:
        embs3d = np.array(embeddingSet)[:, 1]
        embs3d = kp_pad_sequences(embs3d, maxlen=pad_no, value=0,
                                  padding='post', dtype=float)
        embs2d = np.empty((embs3d.shape[0], ), dtype=np.object)
        for i in range(embs3d.shape[0]):
            embs2d[i] = embs3d[i, :, :]
        emb_frame = pd.DataFrame(embs2d, columns=['bertVector1'])

        emb_frame['arg1'] = pd.Series(propositionSet,
                                      index=emb_frame.index)

        dataset = pd.merge(dataset, emb_frame, on='arg1')
        if has_2:
            emb_frame = emb_frame.rename(columns={
                'arg1': 'arg2',
                'bertVector1': 'bertVector2'})
            dataset = pd.merge(dataset, emb_frame, on='arg2')

    if original_sentence_feature:
        original_sentences, _ = get_propositions(dataset,
                                                 column='originalArg1',
                                                 only_props=True)
        embeddingSet = bert_embedding(original_sentences,
                                      filter_spec_tokens=False)

        embs3d = np.array(embeddingSet)[:, 1]
        embs1 = np.array([x[0] for x in embs3d])
        embs2d = np.empty((embs1.shape[0], ), dtype=np.object)
        for i in range(embs1.shape[0]):
            embs2d[i] = embs1[i, :]
        emb_frame = pd.DataFrame(embs2d, columns=['bertOriginalArg1'])

        emb_frame['originalArg1'] = pd.Series(original_sentences,
                                              index=emb_frame.index)

        dataset = pd.merge(dataset, emb_frame, on='originalArg1')
        if has_2:
            emb_frame = \
                emb_frame.rename(columns={
                    'originalArg1': 'originalArg2',
                    'bertOriginalArg1': 'bertOriginalArg2'})
            dataset = pd.merge(dataset, emb_frame, on='originalArg2')

    return dataset


def sentiment_scores(sentence, sid_obj=None):
    """Sentiment analysis of the sentence"""

    if sid_obj is None:
        sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return [sentence, sentiment_dict['neg'], sentiment_dict['neu'],
            sentiment_dict['pos'], sentiment_dict['compound']]


def add_sentiment_scores(dataset, key='arg', has_2=True):
    """Sentiment analysis scores: neg, neu, pos, compound"""

    sid_obj = SentimentIntensityAnalyzer()
    key_t = key.title()
    sentiments = dataset[key + '1'
                         ].drop_duplicates(
                             ).apply(lambda row:
                                     sentiment_scores(row, sid_obj))
    sentiments = pd.DataFrame(sentiments.tolist(),
                              columns=[
                                  key + '1',
                                  'sentNeg' + key_t + '1',
                                  'sentNeu' + key_t + '1',
                                  'sentPos' + key_t + '1',
                                  'sentCompound' + key_t + '1'])
    dataset = pd.merge(dataset, sentiments, on=key + '1')
    if not has_2:
        return dataset

    sentiments = dataset[key + '2'
                         ].drop_duplicates(
                             ).apply(lambda row:
                                     sentiment_scores(row, sid_obj))
    sentiments = pd.DataFrame(sentiments.tolist(),
                              columns=[
                                  key + '2',
                                  'sentNeg' + key_t + '2',
                                  'sentNeu' + key_t + '2',
                                  'sentPos' + key_t + '2',
                                  'sentCompound' + key_t + '2'])
    return pd.merge(dataset, sentiments, on=key + '2')
