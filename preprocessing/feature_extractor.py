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
import bert_embedding

current_dir = os.path.dirname(inspect.stack()[0][1])
WORD2WEC_EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz'
PREMISE_FILE = current_dir + "/premise_indicator.txt"
CLAIM_FILE = current_dir + "/claim_indicator.txt"

def get_propositions(dataset, tokenizer=nltk.tokenize.word_tokenize):
    """Parse propositions
    dataset: the original dataframe
    tokenizer: nltk.tokenize.word_tokenize, bert-embedding.tokenizer, etc.
    Output:
        propositionSet: list of the propositions of the arg1
        parsedPropositions: parsed prop. in the arg1
    """
    
    propositionSet = list(set(dataset['arg1']))
    parsedPropositions = list()

    for proposition in propositionSet:
        words = tokenizer(proposition)
        parsedPropositions.append(nltk.pos_tag(words))

    return propositionSet, parsedPropositions
    

def add_word_vector_feature(dataset, propositionSet, parsedPropositions, word2VecModel=None):
    """Add word2vec feature to the dataset
    """
    if word2VecModel is None:
        word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(WORD2WEC_EMBEDDING_FILE, binary=True)

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
    wordVectorFeature = kp_pad_sequences(wordVectorFeature, value=0, padding='post', dtype=float)

    wordVectorFrame = pd.DataFrame({"arg1": propositionSet, "vector1": wordVectorFeature.tolist()})
    dataset = pd.merge(dataset, wordVectorFrame, on='arg1')
    wordVectorFrame = wordVectorFrame.rename(columns={'arg1':'arg2', "vector1":"vector2"})
    dataset = pd.merge(dataset, wordVectorFrame, on='arg2')

    return dataset

    
def add_pos_feature(dataset, propositionSet, parsedPropositions):
    """Add Part-of-Speech features for every proposition"""

    tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
    lb = LabelBinarizer()
    lb.fit(list(tagdict.keys()))
    
    propositionPOSList = list()

    current = 0
    for proposition in parsedPropositions:

        propositionPOS = get_one_hot_pos(proposition, lb)
        propositionPOSList.append(propositionPOS)

    propositionPOSPadded = kp_pad_sequences(propositionPOSList, value=0, padding='post')
    np.shape(propositionPOSPadded)
    
    posFrame = pd.DataFrame({"arg1":propositionSet, "pos1": propositionPOSPadded.tolist()})
    dataset = pd.merge(dataset, posFrame, on='arg1')
    posFrame = posFrame.rename(columns={'arg1':'arg2', "pos1":"pos2"})
    dataset = pd.merge(dataset, posFrame, on='arg2')

    return dataset

    
def get_one_hot_pos(parsedProposition, label_binarizer):
    """Get one-hot encoded PoS for the proposition"""

    posVectorList = label_binarizer.transform([word[1] for word in parsedProposition])
    posVector = np.array(posVectorList)

    return posVector


def add_keyword_feature(dataset, propositionSet):
    """Add premise and claim flag for every proposition"""

    premise_list = read_key_words(PREMISE_FILE)
    claim_list = read_key_words(CLAIM_FILE)

    keyWordFeatureList = list()

    for proposition in propositionSet:

        originalSentence = dataset.loc[dataset['arg1'] == proposition]['originalArg1'].iloc[0]
        keyWordFeatureList.append(including_keywords_features(proposition, originalSentence, premise_list, claim_list))

    keywordFeatureFrame = pd.DataFrame(data=keyWordFeatureList, columns=["claimIndicatorArg1", "premiseIndicatorArg1"])
    keywordFeatureFrame["arg1"] = propositionSet

    dataset = pd.merge(dataset, keywordFeatureFrame, on='arg1')

    keywordFeatureFrame = keywordFeatureFrame.rename(columns = {'arg1':'arg2', 'claimIndicatorArg1':'claimIndicatorArg2', 'premiseIndicatorArg1': 'premiseIndicatorArg2'})

    return pd.merge(dataset, keywordFeatureFrame, on='arg2')


def including_keywords_features(proposition, original,
                                premise_list, claim_list):
    """Check if the proposition is a keyword or part of a key phrase
    proposition: to check if keyword
    original: sentence
    premise_list: list of premise keywords
    claim_list: list of claim keywords
    Return:
        [premise, claim] - 1 if sentence contains keyword"""

    positionInSentence = original.find(proposition)

    if positionInSentence < 1:

        claim_indicator = check_claim_indicators(original[:len(proposition)], claim_list)
        premise_indicator = check_premise_indicators(original[:len(proposition)], premise_list)

    else:

        wordTokensBefore = nltk.tokenize.word_tokenize(original[:positionInSentence])

        if len(wordTokensBefore) > 1:

            wordsBefore = wordTokensBefore[-2] + wordTokensBefore[-1]

        else:

            wordsBefore = wordTokensBefore[-1]

        extendedSentence = "".join(wordsBefore) + " " + proposition

        claim_indicator = check_claim_indicators(extendedSentence, claim_list)
        premise_indicator = check_premise_indicators(extendedSentence, premise_list)

    return [claim_indicator, premise_indicator]

    
def check_premise_indicators(sentence, premise_list):
    """
    function to detect the presence of argument keywords in a sentence
    :param full sentence:
    :return: 1 if sentence contains keyword
    """

    for indicator in premise_list:
        if re.search(r"\b" + re.escape(indicator) + r"\b", sentence):
            return 1
    return 0


def check_claim_indicators(sentence, claim_list):
    """
    function to detect the presence of argument keywords in a sentence
    :param full sentence:
    :return: True if sentence contains keyword
    """
    for indicator in claim_list:
        if re.search(r"\b" + re.escape(indicator) + r"\b", sentence):
            return 1
    return 0


def read_key_words(file):
    """Reads list of words in file, one keyword per line"""
    return [line.rstrip('\n') for line in open(file)]


def add_token_feature(dataset, propositionSet, parsedPropositions):
    """Add number of propositions in the arguments of the dataset"""

    numberOfTokens = list()

    for i in range(len(propositionSet)):

        numberOfTokens.append([propositionSet[i], len((parsedPropositions[i]))])

    tokenDataFrame = pd.DataFrame(data=numberOfTokens, columns=["proposition", "tokens"])

    tokenDataFrame = tokenDataFrame.rename(columns={'proposition':'arg1', 'tokens':'tokensArg1'})

    dataset = pd.merge(dataset, tokenDataFrame, on='arg1')

    tokenDataFrame = tokenDataFrame.rename(columns={"arg1" : "arg2", "tokensArg1":"tokensArg2"})

    dataset = pd.merge(dataset, tokenDataFrame, on='arg2')

    return dataset

    
def add_shared_noun_feature(dataset, propositionSet, parsedPropositions):
    """Add binary has shared noun and number of shared nouns to the dataset"""
        
    temp = dataset[['arg1','arg2']].apply(lambda row: find_shared_nouns(parsedPropositions[propositionSet.index(row['arg1'])], parsedPropositions[propositionSet.index(row['arg2'])]), axis=1)
    temp = pd.DataFrame(temp.tolist(), columns=['sharedNouns', 'numberOfSharedNouns'])
    dataset["sharedNouns"] = temp.loc[:,'sharedNouns']
    dataset["numberOfSharedNouns"] = temp.loc[:,'numberOfSharedNouns']
        
    return dataset


def find_shared_nouns(proposition, partner):

    arg1Nouns = [word for (word, pos) in proposition if pos == 'NN']
    arg2Nouns = [word for (word, pos) in partner if pos == 'NN']
    intersection = set(arg1Nouns).intersection(arg2Nouns)
    shared = 0
    if len(intersection)>0:
        shared = 1
    
    return [shared, len(intersection)]


def add_same_sentence_feature(dataset):
    """Add binary feature true if the two argument has the same original sentence"""

    dataset["sameSentence"] = dataset[['originalArg1','arg2']].apply(lambda row: int(bool(row['arg2'] in row['originalArg1'])), axis=1)

    return dataset


def add_bert_embeddings(dataset, propositionSet, bert_embedding=None):
    """Add bert embeddings to the dataset. Use matching tokenizer!"""
    if(bert_embedding is None):
        print("Warning! Match tokenizer to have the same propositions!")
        bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    
    embeddingSet = bert_embedding(propositionSet)
    
    embs3d = np.array(embeddingSet)[:,1]
    embs3d = kp_pad_sequences(embs3d, value=0, padding='post', dtype=float)
    np.shape(embs3d)
    embs2d = np.empty((embs3d.shape[0],), dtype=np.object)
    for i in range(embs3d.shape[0]): embs2d[i] = embs3d[i,:,:]
    np.shape(embs2d)
    emb_frame = pd.DataFrame(embs2d, columns=["bert1"])
  
    emb_frame["arg1"] = pd.Series(propositionSet, index=emb_frame.index)
  
    dataset = pd.merge(dataset, emb_frame, on="arg1")
    emb_frame = emb_frame.rename(columns={"arg1":"arg2","bert1":"bert2"})
    dataset = pd.merge(dataset, emb_frame, on="arg2")

    return dataset
